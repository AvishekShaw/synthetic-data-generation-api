#!/usr/bin/env python3
"""
eval_conversation_predictions.py
Two-tier conversation-level evaluation of UserLM predicted conversations.

Reads prediction_metrics_detailed.csv (produced by eval_predictions.py), reconstructs
per-conversation turn sequences for both reference and predicted outputs, then evaluates
them across two tiers:

  Tier 1 — Rule-based (no API calls):
    A. Structural stats       — turn lengths, count, variance
    B. Lexical signals        — hedge rate, certainty/promise rate
    C. First-turn info density — merchant/amount/date front-loading
    D. Breadth-first bundling — concern categories bundled per turn (LLM tell)
    E. Persona consistency    — emotional markers, jargon, style compliance

    For each dimension: computed on BOTH predicted and reference, plus delta (pred − ref).
    Fidelity flags raised when the predicted conversation deviates significantly from reference.

  Tier 2 — LLM-as-judge (requires ANTHROPIC_API_KEY):
    Same 5 dimensions as Tier 1, but framed as FIDELITY evaluation:
    given the reference customer turns, does the predicted conversation
    reproduce the same human-realism properties?
    Score 1–5: 1 = completely divergent from reference behaviour, 5 = indistinguishable.

Usage:
    # Tier 1 only (default, fast, no API)
    python eval_conversation_predictions.py

    # Tier 1 + Tier 2 on all conversations
    python eval_conversation_predictions.py --judge

    # Tier 2 on first 10 conversations only (test)
    python eval_conversation_predictions.py --judge --judge-limit 10

    # Re-run ignoring tier 2 cache
    python eval_conversation_predictions.py --judge --force

Outputs:
    - conversation_predictions_t1.csv   — Tier 1 results per (model_variant, generation_idx)
    - conversation_predictions_t2.csv   — Tier 2 results per (model_variant, generation_idx)
    - images/cp_01_t1_overview.png
    - images/cp_02_t1_deltas.png
    - images/cp_03_t1_by_variant.png
    - images/cp_04_t2_scores.png        (only if --judge)
    - images/cp_05_t2_by_dimension.png  (only if --judge)
"""

import csv
import json
import math
import os
import re
import time
import random
import argparse
import warnings
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import numpy as np
import seaborn as sns

# ── Optional anthropic dependency ────────────────────────────────────────────
_HAS_ANTHROPIC = False
try:
    import anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).parent
DETAILED_CSV  = BASE_DIR / "prediction_metrics_detailed.csv"
T1_CSV        = BASE_DIR / "conversation_predictions_t1.csv"
T2_CSV        = BASE_DIR / "conversation_predictions_t2.csv"
CACHE_DIR     = BASE_DIR / "conv_judge_cache"
IMAGES_DIR    = BASE_DIR / "images"

JUDGE_MODEL      = "claude-sonnet-4-6"
MAX_RETRIES      = 4
RETRY_BASE_SEC   = 2.0
CALL_DELAY_SEC   = 0.5

# Tier 2 dimension weights — must sum to 1.0
DIMENSION_WEIGHTS = {
    "depth_first": 0.25,
    "uncertainty": 0.20,
    "info_drip":   0.15,
    "pragmatic":   0.25,
    "persona":     0.15,
}

ALL_VARIANTS  = ["base_4b", "lora_4b", "base_12b", "lora_12b", "base_27b", "lora_27b"]

# ─────────────────────────────────────────────────────────────────────────────
#  Phrase lists (mirrors eval_conversations.py)
# ─────────────────────────────────────────────────────────────────────────────

HEDGE_PHRASES = [
    "not sure", "i'm not sure", "im not sure", "i think", "i believe",
    "maybe", "perhaps", "possibly", "probably", "might", "could be",
    "i guess", "i don't know", "i dont know", "idk", "honestly",
    "sort of", "kind of", "not certain", "not 100%", "not totally sure",
    "i'm not certain", "im not certain",
]

CERTAINTY_PHRASES = [
    "definitely", "absolutely", "certainly", "for sure", "100%",
    "no question", "without a doubt", "clearly", "obviously", "undoubtedly",
    "no doubt",
]

PROMISE_PHRASES = [
    "i will", "i'll make sure", "i guarantee", "i promise", "i'll ensure",
    "you can count on", "rest assured", "i assure",
]

MILD_FRUSTRATION_MARKERS = [
    "look", "already said", "already told", "already mentioned",
    "already explained", "come on", "seriously", "just want",
    "all i want", "that's why", "thats why",
]

STRONG_FRUSTRATION_MARKERS = [
    "unacceptable", "ridiculous", "outrageous", "i can't believe",
    "i cannot believe", "how is this possible", "this is insane",
    "close my account", "closing my account", "complaint", "cfpb",
    "lawyer", "attorney", "legal", "sue", "escalate", "manager",
    "supervisor", "not acceptable",
]

BANKING_JARGON = [
    "provisional credit", "chargeback", "reg e", "regulation e",
    "billing error", "fraud claim", "dispute resolution",
    "zero liability", "fcba",
]

CONCERN_CATEGORIES = {
    "dispute": [
        "dispute", "didn't make", "did not make", "don't recognize",
        "do not recognize", "didn't authorize", "did not authorize",
        "unauthorized", "fraudulent", "fraud", "charge i", "charge on",
        "transaction i", "not mine", "never made",
    ],
    "card_action": [
        "cancel", "lock", "block", "replace", "new card",
        "card replaced", "shut it down", "close the card",
    ],
    "refund_credit": [
        "money back", "refund", "get it back", "reimburse",
        "provisional credit", "provisional", "credit back",
        "when do i get", "get my money",
    ],
    "timeline_status": [
        "how long", "when will", "how many days", "timeline",
        "status", "update", "what's the eta", "whats the eta",
        "estimated time",
    ],
}

# Pushback phrases — signals user is correcting an agent error (type_b)
PUSHBACK_PHRASES = [
    "that's not right", "that's incorrect", "are you sure", "i thought",
    "i was told", "i've read", "according to", "that doesn't sound right",
    "wait,", "hold on", "actually,", "i don't think that's", "you said earlier",
    "but earlier", "you just said", "that contradicts", "that can't be right",
    "that's wrong", "no, it's", "i believe it's", "reg e", "regulation e",
    "that's not what", "isn't it", "i'm pretty sure", "i read that",
]

# Per-scenario keywords that signal the wrong prior belief is expressed (type_a)
PRIOR_BELIEF_KEYWORDS: dict = {
    0: ["120 days", "120-day", "four months"],
    1: ["description", "rough", "approximate", "don't have the exact"],
    2: ["legally required", "required by law", "mandatory refund", "must refund", "have to refund"],
    3: ["100%", "100 percent", "totally sure", "certain before", "sure before", "positive before"],
    4: ["instantly", "automatically blocks", "immediately blocks", "blocks all pending"],
    5: ["supervisor", "manager", "entitled", "returning caller", "open case"],
    6: ["500", "federal law", "24 hours", "24-hour", "provisional within 24"],
    7: ["police report", "police", "report first", "file a report", "file report"],
}


def _scenario_idx_from_gen_idx(gen_idx, conv_type: str) -> int:
    """Extract scenario index (0–7) from generation_idx."""
    try:
        g = int(gen_idx or 0)
    except (TypeError, ValueError):
        return 0
    if conv_type == "type_b":
        return (g - 2000) % 8
    if conv_type == "type_a":
        return (g - 1000) % 8
    return g % 8


# Role-confusion: assistant-side language in predicted user turns
ROLE_CONFUSION_PHRASES = [
    "i'd be happy to", "i would be happy to", "how can i assist",
    "how can i help", "let me help", "let me assist", "i can help you",
    "i'm here to help", "is there anything else", "feel free to",
    "please don't hesitate", "i understand your concern",
    "i apologize for", "i'm sorry for the inconvenience",
    "thank you for calling", "thank you for contacting",
]

# ─────────────────────────────────────────────────────────────────────────────
#  Visual style
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "base_4b":  "#ADB5BD", "lora_4b":  "#4C9BE8",
    "base_12b": "#CED4DA", "lora_12b": "#2A9D8F",
    "base_27b": "#DEE2E6", "lora_27b": "#E76F51",
    "reference": "#1B3A4B",
    "predicted": "#E9C46A",
}

def _set_style():
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.spines.top":  False,
        "axes.spines.right": False,
        "font.family":      "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.titlesize":   13,
        "axes.labelsize":   11,
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
    })

# ─────────────────────────────────────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReconvConversation:
    """A reconstructed conversation from the prediction CSV."""
    model_variant:    str
    generation_idx:   str
    # Persona (from CSV)
    knowledge_level:      str = ""
    emotional_state:      str = ""
    communication_style:  str = ""
    goal_clarity:         str = ""
    # Scenario (from CSV)
    certainty:               str = ""
    information_completeness: str = ""
    merchant_name:            str = ""
    amount:                   str = ""
    transaction_date:         str = ""
    expected_resolution:      str = ""
    # Conversation type + probe metadata (from CSV)
    conversation_type:    str = "success"   # success | type_a | type_b
    wrong_prior_belief:   str = ""          # type_a: the prior belief sentence
    agent_failure_mode:   str = ""          # type_b: planted error category
    user_caught_error:    object = None     # type_b: bool
    # Ordered customer turns (text only)
    predicted_turns: list = field(default_factory=list)
    reference_turns: list = field(default_factory=list)
    # Turn count
    n_turns: int = 0


@dataclass
class T1TurnStats:
    """Tier 1 stats computed for one turn sequence (predicted or reference)."""
    n_turns:        int   = 0
    words_mean:     float = 0.0
    words_std:      float = 0.0
    words_p10:      float = 0.0
    words_p90:      float = 0.0
    hedge_rate:     float = 0.0
    certainty_rate: float = 0.0
    promise_rate:   float = 0.0
    hedge_surplus:  float = 0.0   # hedge_rate - certainty_rate
    facts_in_turn1: int   = 0
    info_density:   float = 0.0
    max_cats_per_turn:      int = 0
    turns_with_multi_cat:   int = 0
    breadth_first_flag:     bool = False
    frustration_present:    bool = False
    jargon_present:         bool = False
    role_confused_turns:    int  = 0
    # Probe-type signals
    pushback_turn_count:     int   = 0    # type_b: turns with pushback language
    pushback_first_turn_idx: int   = -1   # type_b: first pushback turn index (-1 = none)
    prior_belief_rate:       float = 0.0  # type_a: fraction of turns expressing prior belief


@dataclass
class T1Result:
    """Tier 1 result for one (model_variant, generation_idx) pair."""
    model_variant:    str
    model_size:       str
    is_lora:          bool
    generation_idx:   str
    # Metadata
    knowledge_level:      str = ""
    emotional_state:      str = ""
    communication_style:  str = ""
    goal_clarity:         str = ""
    certainty:            str = ""
    information_completeness: str = ""
    n_turns: int = 0
    # Conversation type + probe metadata
    conversation_type:  str = "success"
    wrong_prior_belief: str = ""
    agent_failure_mode: str = ""

    # ── Predicted stats ──────────────────────────────────────────────────────
    pred_words_mean:    float = 0.0
    pred_words_std:     float = 0.0
    pred_words_p10:     float = 0.0
    pred_words_p90:     float = 0.0
    pred_hedge_rate:    float = 0.0
    pred_certainty_rate: float = 0.0
    pred_promise_rate:  float = 0.0
    pred_hedge_surplus: float = 0.0
    pred_facts_turn1:   int   = 0
    pred_info_density:  float = 0.0
    pred_max_cats:      int   = 0
    pred_multi_cat_turns: int = 0
    pred_breadth_first: bool  = False
    pred_frustration:   bool  = False
    pred_jargon:        bool  = False
    pred_role_confused_turns: int = 0
    # Probe signals — predicted
    pred_pushback_count:     int   = 0
    pred_pushback_first_idx: int   = -1
    pred_prior_belief_rate:  float = 0.0

    # ── Reference stats ──────────────────────────────────────────────────────
    ref_words_mean:    float = 0.0
    ref_words_std:     float = 0.0
    ref_words_p10:     float = 0.0
    ref_words_p90:     float = 0.0
    ref_hedge_rate:    float = 0.0
    ref_certainty_rate: float = 0.0
    ref_promise_rate:  float = 0.0
    ref_hedge_surplus: float = 0.0
    ref_facts_turn1:   int   = 0
    ref_info_density:  float = 0.0
    ref_max_cats:      int   = 0
    ref_multi_cat_turns: int = 0
    ref_breadth_first: bool  = False
    ref_frustration:   bool  = False
    ref_jargon:        bool  = False
    # Probe signals — reference
    ref_pushback_count:     int   = 0
    ref_pushback_first_idx: int   = -1
    ref_prior_belief_rate:  float = 0.0

    # ── Delta (pred − ref) ───────────────────────────────────────────────────
    delta_words_mean:    float = 0.0
    delta_words_std:     float = 0.0
    delta_hedge_rate:    float = 0.0
    delta_certainty_rate: float = 0.0
    delta_promise_rate:  float = 0.0
    delta_hedge_surplus: float = 0.0
    delta_info_density:  float = 0.0

    # ── Fidelity flags ────────────────────────────────────────────────────────
    # Raised when predicted deviates meaningfully from reference
    flag_length_inflated:   bool = False   # pred turns >30% longer than ref on avg
    flag_length_deflated:   bool = False   # pred turns >30% shorter than ref on avg
    flag_over_certain:      bool = False   # certainty_rate pred >> ref by >0.2
    flag_under_hedge:       bool = False   # hedge_rate pred << ref by >0.2
    flag_info_front_loaded: bool = False   # pred front-loads more facts than ref
    flag_breadth_first_new: bool = False   # pred shows breadth-first but ref doesn't
    flag_role_confused:     bool = False   # any role confusion in predicted turns
    flag_persona_jargon:    bool = False   # novice predicted jargon not in reference
    # Probe-type flags
    flag_missed_pushback:       bool = False  # type_b: ref pushes back, pred doesn't
    flag_wrong_belief_missing:  bool = False  # type_a: ref expresses prior belief, pred doesn't
    fidelity_flag_count:        int  = 0


@dataclass
class T2Result:
    """Tier 2 LLM-as-judge result for one (model_variant, generation_idx) pair."""
    model_variant:  str
    model_size:     str
    is_lora:        bool
    generation_idx: str
    knowledge_level:     str = ""
    emotional_state:     str = ""
    communication_style: str = ""
    goal_clarity:        str = ""
    certainty:           str = ""
    conversation_type:   str = "success"
    # Fidelity scores 1–5 per dimension
    # Shared across all types:
    depth_first_score:   float = 0.0
    uncertainty_score:   float = 0.0
    pragmatic_score:     float = 0.0
    persona_score:       float = 0.0
    overall_score:       float = 0.0
    # success + type_a only:
    info_drip_score:     float = 0.0
    # type_a specific:
    prior_belief_score:      float = 0.0   # did user express + update prior belief?
    info_incompleteness_score: float = 0.0 # incomplete info revealed at right rate?
    # type_b specific:
    error_detection_score:   float = 0.0   # did user catch the planted error?
    pushback_calibration_score: float = 0.0  # was pushback intensity persona-appropriate?
    # Rationales
    depth_first_rationale: str = ""
    uncertainty_rationale: str = ""
    info_drip_rationale:   str = ""
    pragmatic_rationale:   str = ""
    persona_rationale:     str = ""
    prior_belief_rationale:    str = ""
    error_detection_rationale: str = ""
    pushback_calibration_rationale: str = ""
    standout_divergence:   str = ""
    standout_match:        str = ""
    # Metadata
    model_used:  str = ""
    parse_error: str = ""
    cached:      bool = False

# ─────────────────────────────────────────────────────────────────────────────
#  Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def word_count(text: str) -> int:
    return len(text.split()) if text.strip() else 0

def safe_mean(vals: list) -> float:
    return sum(vals) / len(vals) if vals else 0.0

def safe_std(vals: list) -> float:
    if len(vals) < 2:
        return 0.0
    mean = safe_mean(vals)
    return math.sqrt(sum((x - mean) ** 2 for x in vals) / len(vals))

def percentile(data: list, p: float) -> float:
    if not data:
        return 0.0
    ds = sorted(data)
    idx = (len(ds) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(ds) - 1)
    return ds[lo] + (ds[hi] - ds[lo]) * (idx - lo)

def count_phrases(text_lower: str, phrases: list) -> int:
    return sum(1 for p in phrases if p in text_lower)

def extract_amount_digits(s: str) -> str:
    return re.sub(r"[^\d.]", "", s) if s else ""

def extract_date_tokens(s: str) -> tuple[str, str]:
    """Return (day_digits, month_name_lower)."""
    day = re.search(r"(\d+)", s)
    month = re.match(r"([a-zA-Z]+)", s.strip())
    return (day.group(1) if day else ""), (month.group(1).lower() if month else "")

def strip_eos(text: str) -> str:
    return re.sub(r"\s*<eos>\s*$", "", text, flags=re.IGNORECASE).strip()

# ─────────────────────────────────────────────────────────────────────────────
#  Data loading & conversation reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def load_detailed_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))

def reconstruct_conversations(rows: list[dict]) -> dict[tuple[str, str], ReconvConversation]:
    """
    Group non-EOS rows by (model_variant, generation_idx), ordered by example_idx.
    Returns {(model_variant, generation_idx): ReconvConversation}.
    """
    print("  → Reconstructing conversations from CSV rows …")

    # Filter to text-only turns (exclude EOS turns — no content to evaluate)
    text_rows = [r for r in rows if str(r.get("is_eos_example", "")).strip().lower() not in ("true", "1")]

    # Sort by (model_variant, generation_idx, example_idx)
    try:
        text_rows.sort(key=lambda r: (
            r["model_variant"],
            str(r.get("generation_idx", "")),
            int(r.get("example_idx", 0)),
        ))
    except (ValueError, TypeError):
        text_rows.sort(key=lambda r: (
            r["model_variant"],
            str(r.get("generation_idx", "")),
            str(r.get("example_idx", "")),
        ))

    convs: dict[tuple[str, str], ReconvConversation] = {}

    for row in text_rows:
        variant = row["model_variant"]
        gen_idx = str(row.get("generation_idx", ""))
        key = (variant, gen_idx)

        if key not in convs:
            # Derive conversation_type: prefer explicit CSV column, fall back to gen_idx range
            ctype = row.get("conversation_type", "")
            if not ctype:
                try:
                    g = int(gen_idx)
                    ctype = "type_b" if g >= 2000 else ("type_a" if g >= 1000 else "success")
                except (ValueError, TypeError):
                    ctype = "success"

            # user_caught_error may be stored as string "True"/"False"
            uce_raw = row.get("user_caught_error", "")
            if isinstance(uce_raw, bool):
                uce = uce_raw
            elif str(uce_raw).lower() == "true":
                uce = True
            elif str(uce_raw).lower() == "false":
                uce = False
            else:
                uce = None

            convs[key] = ReconvConversation(
                model_variant   = variant,
                generation_idx  = gen_idx,
                knowledge_level = row.get("knowledge_level", ""),
                emotional_state = row.get("emotional_state", ""),
                communication_style = row.get("communication_style", ""),
                goal_clarity    = row.get("goal_clarity", ""),
                certainty       = row.get("certainty", ""),
                information_completeness = row.get("information_completeness", ""),
                merchant_name   = row.get("merchant_name", ""),
                amount          = row.get("amount", ""),
                transaction_date = row.get("transaction_date", ""),
                expected_resolution = row.get("expected_resolution", ""),
                conversation_type   = ctype,
                wrong_prior_belief  = row.get("wrong_prior_belief", ""),
                agent_failure_mode  = row.get("agent_failure_mode", ""),
                user_caught_error   = uce,
            )

        pred = strip_eos(row.get("predicted_output", ""))
        ref  = strip_eos(row.get("expected_output",  ""))

        if pred:
            convs[key].predicted_turns.append(pred)
        if ref:
            convs[key].reference_turns.append(ref)

    for conv in convs.values():
        conv.n_turns = max(len(conv.predicted_turns), len(conv.reference_turns))

    print(f"  → {len(convs)} (model_variant, conversation) pairs reconstructed")
    return convs

# ─────────────────────────────────────────────────────────────────────────────
#  Tier 1 — Rule-based evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_turn_stats(
    turns: list[str],
    scenario: dict,
    check_role_confusion: bool = False,
    conversation_type: str = "success",
    scenario_idx: int = 0,
) -> T1TurnStats:
    """Compute all Tier 1 stats for one turn sequence."""
    stats = T1TurnStats(n_turns=len(turns))
    if not turns:
        return stats

    wc_list = [word_count(t) for t in turns]
    stats.words_mean = round(safe_mean(wc_list), 2)
    stats.words_std  = round(safe_std(wc_list), 2)
    stats.words_p10  = round(percentile(wc_list, 10), 2)
    stats.words_p90  = round(percentile(wc_list, 90), 2)

    total_hedge, total_certain, total_promise = 0, 0, 0
    for turn in turns:
        tl = turn.lower()
        total_hedge   += count_phrases(tl, HEDGE_PHRASES)
        total_certain += count_phrases(tl, CERTAINTY_PHRASES)
        total_promise += count_phrases(tl, PROMISE_PHRASES)

    n = len(turns)
    stats.hedge_rate     = round(total_hedge   / n, 3)
    stats.certainty_rate = round(total_certain / n, 3)
    stats.promise_rate   = round(total_promise / n, 3)
    stats.hedge_surplus  = round(stats.hedge_rate - stats.certainty_rate, 3)

    # C. Info density in first turn
    if turns:
        t1 = turns[0].lower()
        merchant = scenario.get("merchant_name", "")
        amount   = scenario.get("amount", "")
        date_str = scenario.get("transaction_date", "")

        merchant_hit = bool(merchant and merchant.lower() in t1)
        amt_digits   = extract_amount_digits(amount)
        amount_hit   = bool(amt_digits and amt_digits in t1.replace(",", ""))
        day, month   = extract_date_tokens(date_str)
        date_hit     = bool((day and day in t1) or (month and month in t1))

        facts = sum([merchant_hit, amount_hit, date_hit])
        stats.facts_in_turn1 = facts
        stats.info_density   = round(facts / 3, 3)

    # D. Breadth-first bundling
    max_cats = 0
    multi_cat_turns = 0
    for turn in turns:
        tl = turn.lower()
        cats_hit = sum(
            1 for cat, phrases in CONCERN_CATEGORIES.items()
            if any(p in tl for p in phrases)
        )
        max_cats = max(max_cats, cats_hit)
        if cats_hit >= 2:
            multi_cat_turns += 1

    stats.max_cats_per_turn    = max_cats
    stats.turns_with_multi_cat = multi_cat_turns
    stats.breadth_first_flag   = (multi_cat_turns >= 2 or max_cats >= 3)

    # E. Persona signals
    all_text = " ".join(turns).lower()
    frust_markers = (
        MILD_FRUSTRATION_MARKERS + STRONG_FRUSTRATION_MARKERS
    )
    stats.frustration_present = any(p in all_text for p in frust_markers)
    stats.jargon_present      = any(p in all_text for p in BANKING_JARGON)

    # Role confusion (predicted turns only)
    if check_role_confusion:
        rc = 0
        for turn in turns:
            tl = turn.lower()
            if (re.match(r"^\s*agent:", tl) or
                    any(p in tl for p in ROLE_CONFUSION_PHRASES)):
                rc += 1
        stats.role_confused_turns = rc

    # F. Probe-type signals
    if conversation_type == "type_b":
        pb_count = 0
        pb_first = -1
        for i, turn in enumerate(turns):
            tl = turn.lower()
            if any(p in tl for p in PUSHBACK_PHRASES):
                pb_count += 1
                if pb_first == -1:
                    pb_first = i
        stats.pushback_turn_count     = pb_count
        stats.pushback_first_turn_idx = pb_first

    elif conversation_type == "type_a":
        keywords = PRIOR_BELIEF_KEYWORDS.get(scenario_idx % 8, [])
        if keywords and turns:
            hits = sum(
                1 for turn in turns
                if any(k in turn.lower() for k in keywords)
            )
            stats.prior_belief_rate = round(hits / len(turns), 3)

    return stats


def run_tier1(convs: dict) -> list[T1Result]:
    """Run Tier 1 on all reconstructed conversations."""
    print("\n[TIER 1] Running rule-based evaluation …")
    results = []
    total = len(convs)

    for i, ((variant, gen_idx), conv) in enumerate(convs.items(), 1):
        if i % 20 == 0 or i == total:
            print(f"  → {i}/{total} conversations processed …")

        size = variant.split("_", 1)[1] if "_" in variant else variant
        is_lora = variant.startswith("lora_")
        ctype = conv.conversation_type
        sc_idx = _scenario_idx_from_gen_idx(conv.generation_idx, ctype)

        scenario = {
            "merchant_name":   conv.merchant_name,
            "amount":          conv.amount,
            "transaction_date": conv.transaction_date,
        }

        pred_stats = _compute_turn_stats(
            conv.predicted_turns, scenario,
            check_role_confusion=True,
            conversation_type=ctype, scenario_idx=sc_idx,
        )
        ref_stats = _compute_turn_stats(
            conv.reference_turns, scenario,
            check_role_confusion=False,
            conversation_type=ctype, scenario_idx=sc_idx,
        )

        r = T1Result(
            model_variant   = variant,
            model_size      = size,
            is_lora         = is_lora,
            generation_idx  = gen_idx,
            knowledge_level = conv.knowledge_level,
            emotional_state = conv.emotional_state,
            communication_style = conv.communication_style,
            goal_clarity    = conv.goal_clarity,
            certainty       = conv.certainty,
            information_completeness = conv.information_completeness,
            n_turns         = conv.n_turns,
            conversation_type  = ctype,
            wrong_prior_belief = conv.wrong_prior_belief,
            agent_failure_mode = conv.agent_failure_mode,
            # Predicted
            pred_words_mean    = pred_stats.words_mean,
            pred_words_std     = pred_stats.words_std,
            pred_words_p10     = pred_stats.words_p10,
            pred_words_p90     = pred_stats.words_p90,
            pred_hedge_rate    = pred_stats.hedge_rate,
            pred_certainty_rate = pred_stats.certainty_rate,
            pred_promise_rate  = pred_stats.promise_rate,
            pred_hedge_surplus = pred_stats.hedge_surplus,
            pred_facts_turn1   = pred_stats.facts_in_turn1,
            pred_info_density  = pred_stats.info_density,
            pred_max_cats      = pred_stats.max_cats_per_turn,
            pred_multi_cat_turns = pred_stats.turns_with_multi_cat,
            pred_breadth_first = pred_stats.breadth_first_flag,
            pred_frustration   = pred_stats.frustration_present,
            pred_jargon        = pred_stats.jargon_present,
            pred_role_confused_turns = pred_stats.role_confused_turns,
            pred_pushback_count      = pred_stats.pushback_turn_count,
            pred_pushback_first_idx  = pred_stats.pushback_first_turn_idx,
            pred_prior_belief_rate   = pred_stats.prior_belief_rate,
            # Reference
            ref_words_mean    = ref_stats.words_mean,
            ref_words_std     = ref_stats.words_std,
            ref_words_p10     = ref_stats.words_p10,
            ref_words_p90     = ref_stats.words_p90,
            ref_hedge_rate    = ref_stats.hedge_rate,
            ref_certainty_rate = ref_stats.certainty_rate,
            ref_promise_rate  = ref_stats.promise_rate,
            ref_hedge_surplus = ref_stats.hedge_surplus,
            ref_facts_turn1   = ref_stats.facts_in_turn1,
            ref_info_density  = ref_stats.info_density,
            ref_max_cats      = ref_stats.max_cats_per_turn,
            ref_multi_cat_turns = ref_stats.turns_with_multi_cat,
            ref_breadth_first = ref_stats.breadth_first_flag,
            ref_frustration   = ref_stats.frustration_present,
            ref_jargon        = ref_stats.jargon_present,
            ref_pushback_count      = ref_stats.pushback_turn_count,
            ref_pushback_first_idx  = ref_stats.pushback_first_turn_idx,
            ref_prior_belief_rate   = ref_stats.prior_belief_rate,
        )

        # Deltas (pred − ref)
        r.delta_words_mean    = round(r.pred_words_mean    - r.ref_words_mean,    3)
        r.delta_words_std     = round(r.pred_words_std     - r.ref_words_std,     3)
        r.delta_hedge_rate    = round(r.pred_hedge_rate    - r.ref_hedge_rate,    3)
        r.delta_certainty_rate = round(r.pred_certainty_rate - r.ref_certainty_rate, 3)
        r.delta_promise_rate  = round(r.pred_promise_rate  - r.ref_promise_rate,  3)
        r.delta_hedge_surplus = round(r.pred_hedge_surplus - r.ref_hedge_surplus, 3)
        r.delta_info_density  = round(r.pred_info_density  - r.ref_info_density,  3)

        # Fidelity flags
        # Thresholds tuned for short conversational turns (avg 8–15 words):
        #   length: flag if avg turn is >15% longer/shorter than reference
        #   lexical: flag if abs delta > 0.10 per turn (1 phrase per 10 turns)
        #   info density: flag if pred volunteers meaningfully more facts in T1
        #
        # Type-aware suppression:
        #   type_b: over_certain and under_hedge are suppressed because pushback
        #           language legitimately raises certainty and lowers hedging.
        flags = 0
        if r.ref_words_mean > 0:
            ratio = r.pred_words_mean / r.ref_words_mean
            if ratio > 1.15:
                r.flag_length_inflated = True; flags += 1
            if ratio < 0.85:
                r.flag_length_deflated = True; flags += 1

        # Suppress certainty/hedge flags for type_b — assertive corrective language
        # is expected and should not be counted as a fidelity failure.
        if ctype != "type_b":
            if r.delta_certainty_rate > 0.10:
                r.flag_over_certain = True; flags += 1
            if r.delta_hedge_rate < -0.10:
                r.flag_under_hedge = True; flags += 1

        if r.pred_info_density > r.ref_info_density + 0.15:
            r.flag_info_front_loaded = True; flags += 1
        if r.pred_breadth_first and not r.ref_breadth_first:
            r.flag_breadth_first_new = True; flags += 1
        if r.pred_role_confused_turns > 0:
            r.flag_role_confused = True; flags += 1
        if (r.pred_jargon and not r.ref_jargon
                and conv.knowledge_level in ("novice", "intermediate")):
            r.flag_persona_jargon = True; flags += 1

        # Probe-type fidelity flags
        if ctype == "type_b":
            # Critical: reference had pushback but predicted has none
            if r.ref_pushback_count > 0 and r.pred_pushback_count == 0:
                r.flag_missed_pushback = True; flags += 1

        if ctype == "type_a":
            # Reference expressed prior belief but predicted never did
            if r.ref_prior_belief_rate > 0.0 and r.pred_prior_belief_rate == 0.0:
                r.flag_wrong_belief_missing = True; flags += 1

        r.fidelity_flag_count = flags
        results.append(r)

    print(f"  ✓ Tier 1 complete: {len(results)} results")
    return results


def save_t1_csv(results: list[T1Result], path: Path):
    if not results:
        return
    rows = [asdict(r) for r in results]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✓ {path.name} written ({len(rows)} rows)")

# ─────────────────────────────────────────────────────────────────────────────
#  Tier 2 — LLM-as-judge (fidelity framing)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_T2 = """You are an expert evaluator for user-simulation language models (UserLMs) \
in the banking and financial services domain.

Your task is to evaluate whether a PREDICTED set of customer turns faithfully reproduces \
the human-realism properties of the REFERENCE customer turns in a banking dispute conversation.

Both sequences come from the same conversation scenario — the predicted turns were generated \
by a fine-tuned language model trained to simulate the customer side of the conversation.
The reference turns are the synthetic ground-truth customer responses used during training.

You evaluate FIDELITY: does the predicted conversation exhibit the same human-like properties \
as the reference? You are grounded in:

- Wang et al. (2025): Real humans use depth-first questioning and appropriate uncertainty \
expression. LLM-generated users over-commit and bundle multiple concerns.
- Mannekote et al. (2025): Pragmatic naturalness means fitting the conversational context.
- Chen et al. (2024): Persona attributes must visibly manifest in the dialogue."""


def _conv_header(conv: "ReconvConversation") -> str:
    """Shared header block used by all three judge prompts."""
    ref_transcript  = "\n".join(
        f"[REFERENCE TURN {i+1}]: {t}"
        for i, t in enumerate(conv.reference_turns)
    )
    pred_transcript = "\n".join(
        f"[PREDICTED TURN {i+1}]: {t}"
        for i, t in enumerate(conv.predicted_turns)
    )
    return (
        f"━━━━━━━━━━━━━━━━━━━━━━━━\nPERSONA\n━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"knowledge_level:      {conv.knowledge_level}\n"
        f"emotional_state:      {conv.emotional_state}\n"
        f"communication_style:  {conv.communication_style}\n"
        f"goal_clarity:         {conv.goal_clarity}\n\n"
        f"SCENARIO\n━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"certainty:                 {conv.certainty}\n"
        f"information_completeness:  {conv.information_completeness}\n"
        f"merchant:                  {conv.merchant_name}\n"
        f"amount:                    {conv.amount}\n"
        f"transaction_date:          {conv.transaction_date}\n"
        f"expected_resolution:       {conv.expected_resolution}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\nREFERENCE CUSTOMER TURNS\n━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{ref_transcript}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\nPREDICTED CUSTOMER TURNS\n━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{pred_transcript}"
    )


def build_judge_prompt_type_a(conv: "ReconvConversation") -> str:
    """Tier 2 judge prompt for type_a (inadvertent probing) conversations."""
    header = _conv_header(conv)
    return f"""Below is a TYPE A (inadvertent probing) banking dispute conversation.
The customer holds a wrong prior belief: {conv.wrong_prior_belief or "(see conversation)"}
The customer also has information_completeness = {conv.information_completeness}.

Your task is to evaluate whether the PREDICTED customer turns faithfully reproduce
the human-realism properties of the REFERENCE customer turns.

{header}

━━━━━━━━━━━━━━━━━━━━━━━━
SCORING RUBRIC
━━━━━━━━━━━━━━━━━━━━━━━━

Score each dimension 1–5 (1=completely divergent, 3=partial match, 5=faithfully reproduced):

DIMENSION 1 — DEPTH-FIRST QUESTIONING FIDELITY
  Does the predicted sequence match the reference in how concerns are raised?
  5: Same sequencing — one concern at a time, same as reference.
  3: Mostly matches but one turn bundles concerns the reference spread out.
  1: Predicted front-loads multiple concerns that reference raised sequentially.

DIMENSION 2 — UNCERTAINTY EXPRESSION FIDELITY
  Does the predicted sequence match the reference's hedge/commitment balance?
  5: Indistinguishable hedge/certainty balance from reference.
  3: Slightly more or less certain than reference but same general register.
  1: Systematically more committed or more uncertain than reference.

DIMENSION 3 — PRIOR BELIEF PERSISTENCE FIDELITY
  Does the predicted sequence naturally express the wrong prior belief the same way
  the reference does, and update (or not) when the agent provides correct information?
  5: Prior belief expressed at same turns, with same persistence/update pattern as reference.
  3: Some belief expression but weaker, earlier, or misplaced compared to reference.
  1: Predicted never expresses the prior belief that the reference clearly holds and voices.

DIMENSION 4 — INCOMPLETE INFO REVELATION FIDELITY
  Given information_completeness = {conv.information_completeness}, does the predicted
  sequence reveal facts at the same incomplete/partial rate as the reference?
  5: Information revealed (or withheld) in same turns and order as reference.
  3: Key facts appear but in different turns or order than reference.
  1: Predicted front-loads facts when reference dripped them, or vice versa.

DIMENSION 5 — PERSONA FIDELITY
  Does the predicted sequence manifest the persona as strongly as the reference?
  5: Emotional state ({conv.emotional_state}), knowledge ({conv.knowledge_level}),
     and style ({conv.communication_style}) all as visible as in the reference.
  3: Some persona attributes present but one dimension weaker than reference.
  1: Predicted has generic tone that does not match reference persona expression.

━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY a valid JSON object. No markdown, no commentary outside JSON.

{{
  "depth_first_score": <integer 1-5>,
  "depth_first_rationale": "<one sentence>",
  "uncertainty_score": <integer 1-5>,
  "uncertainty_rationale": "<one sentence>",
  "prior_belief_score": <integer 1-5>,
  "prior_belief_rationale": "<one sentence comparing a specific turn>",
  "info_incompleteness_score": <integer 1-5>,
  "info_incompleteness_rationale": "<one sentence>",
  "persona_score": <integer 1-5>,
  "persona_rationale": "<one sentence>",
  "standout_divergence": "<the single biggest way the predicted conversation diverges>",
  "standout_match": "<the single property reproduced most faithfully>"
}}"""


def build_judge_prompt_type_b(conv: "ReconvConversation") -> str:
    """Tier 2 judge prompt for type_b (adversarial probing) conversations."""
    header = _conv_header(conv)
    return f"""Below is a TYPE B (adversarial probing) banking dispute conversation.
The bank agent made a planted error: {conv.agent_failure_mode or "(see conversation)"}.

Your task is to evaluate whether the PREDICTED customer turns faithfully reproduce
the human-realism properties of the REFERENCE customer turns, with special focus on
whether the user catches and responds to the agent's error.

{header}

━━━━━━━━━━━━━━━━━━━━━━━━
SCORING RUBRIC
━━━━━━━━━━━━━━━━━━━━━━━━

Score each dimension 1–5 (1=completely divergent, 3=partial match, 5=faithfully reproduced):

DIMENSION 1 — ERROR DETECTION FIDELITY
  Does the predicted sequence catch the planted agent error ({conv.agent_failure_mode})
  at approximately the same turn and with the same intensity as the reference?
  5: Predicted catches the error at the same turn with comparable pushback intensity.
  3: Error is noticed but at a different turn or with noticeably different intensity.
  1: Predicted completely misses the error that the reference clearly caught and pushed back on.

DIMENSION 2 — UNCERTAINTY EXPRESSION FIDELITY
  Does the predicted sequence match the reference's hedge/commitment balance?
  Note: pushback language is EXPECTED here — assertive certainty is appropriate.
  5: Certainty/assertiveness in predicted matches the reference's pushback register.
  3: Slightly more passive or aggressive than reference on correction turns.
  1: Predicted is systematically more deferential than reference (doesn't push back).

DIMENSION 3 — PUSHBACK CALIBRATION FIDELITY
  Is the pushback tone proportional to the persona ({conv.emotional_state},
  {conv.knowledge_level}) the same way it is in the reference?
  5: Pushback intensity and register perfectly match persona and reference.
  3: Pushback present but tone is off — e.g., too aggressive for a calm persona
     or too mild for an escalating expert.
  1: Pushback tone is completely mismatched to persona compared to reference.

DIMENSION 4 — PRAGMATIC NATURALNESS FIDELITY
  Do the predicted turns sound as natural and colloquial as the reference?
  5: Predicted turns are equally natural — no turns more scripted than reference.
  3: 1–2 predicted turns feel more formal or scripted than reference.
  1: Multiple predicted turns are noticeably more robotic or templated than reference.

DIMENSION 5 — PERSONA FIDELITY
  Does the predicted sequence manifest the persona as strongly as the reference?
  5: Emotional state ({conv.emotional_state}), knowledge ({conv.knowledge_level}),
     and style ({conv.communication_style}) all as visible as in the reference.
  3: Some persona attributes present but one dimension weaker than reference.
  1: Predicted has generic tone that does not match reference persona expression.

━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY a valid JSON object. No markdown, no commentary outside JSON.

{{
  "error_detection_score": <integer 1-5>,
  "error_detection_rationale": "<one sentence citing the specific error turn>",
  "uncertainty_score": <integer 1-5>,
  "uncertainty_rationale": "<one sentence>",
  "pushback_calibration_score": <integer 1-5>,
  "pushback_calibration_rationale": "<one sentence comparing tone to persona>",
  "pragmatic_score": <integer 1-5>,
  "pragmatic_rationale": "<one sentence>",
  "persona_score": <integer 1-5>,
  "persona_rationale": "<one sentence>",
  "standout_divergence": "<the single biggest way the predicted conversation diverges>",
  "standout_match": "<the single property reproduced most faithfully>"
}}"""


def build_judge_prompt(conv: "ReconvConversation") -> str:
    """Route to the correct judge prompt based on conversation type."""
    if conv.conversation_type == "type_a":
        return build_judge_prompt_type_a(conv)
    if conv.conversation_type == "type_b":
        return build_judge_prompt_type_b(conv)
    # Default: success conversations
    return _build_judge_prompt_success(conv)


def _build_judge_prompt_success(conv: "ReconvConversation") -> str:
    """Original Tier 2 judge prompt for success conversations."""
    header = _conv_header(conv)
    return f"""Below is the persona and scenario for a banking dispute conversation, \
followed by the REFERENCE customer turns (ground truth) and the PREDICTED customer turns \
(generated by a UserLM model). Evaluate how faithfully the predicted turns reproduce \
the human-realism properties of the reference.

{header}

━━━━━━━━━━━━━━━━━━━━━━━━
SCORING RUBRIC
━━━━━━━━━━━━━━━━━━━━━━━━

Score each dimension 1–5:
  1 = predicted completely diverges from reference behaviour on this dimension
  3 = partial match — some properties reproduced but noticeable differences
  5 = predicted faithfully reproduces the reference on this dimension

DIMENSION 1 — DEPTH-FIRST QUESTIONING FIDELITY
  Does the predicted sequence match the reference in how concerns are raised?
  5: Same sequencing of concerns — one at a time, same as reference.
  3: Mostly matches but one turn bundles concerns the reference spread out.
  1: Predicted front-loads multiple concerns that reference raised sequentially.

DIMENSION 2 — UNCERTAINTY EXPRESSION FIDELITY
  Does the predicted sequence match the reference's level of hedging and commitment language?
  5: Hedge/certainty balance in predicted is indistinguishable from reference.
  3: Slightly more or less certain than reference, but same general register.
  1: Predicted is systematically more committed or more uncertain than reference.

DIMENSION 3 — INFORMATION DRIP FIDELITY
  Does the predicted sequence reveal {conv.merchant_name} / {conv.amount} / \
{conv.transaction_date} at the same rate as the reference?
  5: Information revealed in same turns and same order as reference.
  3: Key facts appear but in different turns or order than reference.
  1: Predicted front-loads all facts when reference dripped them, or vice versa.

DIMENSION 4 — PRAGMATIC NATURALNESS FIDELITY
  Do the predicted turns sound as natural and colloquial as the reference?
  5: Predicted turns are equally natural — no turns sound more scripted than reference.
  3: 1–2 predicted turns feel slightly more formal or scripted than equivalent reference turns.
  1: Multiple predicted turns are noticeably more robotic or templated than reference.

DIMENSION 5 — PERSONA FIDELITY
  Does the predicted sequence manifest the assigned persona as strongly as the reference?
  5: Emotional state ({conv.emotional_state}), knowledge ({conv.knowledge_level}), \
and style ({conv.communication_style}) are all as clearly visible as in the reference.
  3: Some persona attributes present but one dimension is weaker than reference.
  1: Predicted has generic tone that does not match the reference's persona expression.

━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━

Return ONLY a valid JSON object with exactly these keys. No markdown, no explanation outside JSON.

{{
  "depth_first_score": <integer 1-5>,
  "depth_first_rationale": "<one sentence comparing a specific predicted turn to its reference equivalent>",
  "uncertainty_score": <integer 1-5>,
  "uncertainty_rationale": "<one sentence>",
  "info_drip_score": <integer 1-5>,
  "info_drip_rationale": "<one sentence>",
  "pragmatic_score": <integer 1-5>,
  "pragmatic_rationale": "<one sentence>",
  "persona_score": <integer 1-5>,
  "persona_rationale": "<one sentence>",
  "standout_divergence": "<the single biggest way the predicted conversation diverges from reference, or empty string>",
  "standout_match": "<the single property the predicted conversation reproduces most faithfully, or empty string>"
}}"""


def cache_path_t2(variant: str, gen_idx: str) -> Path:
    safe_variant  = re.sub(r"[^\w]", "_", variant)
    safe_gen_idx  = re.sub(r"[^\w]", "_", str(gen_idx))
    return CACHE_DIR / f"{safe_variant}__{safe_gen_idx}.json"


def load_cache_t2(variant: str, gen_idx: str) -> Optional[dict]:
    p = cache_path_t2(variant, gen_idx)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    return None


def save_cache_t2(variant: str, gen_idx: str, data: dict):
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path_t2(variant, gen_idx).write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def call_judge_t2(client, conv: ReconvConversation, model: str) -> dict:
    prompt = build_judge_prompt(conv)
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=600,
                system=SYSTEM_PROMPT_T2,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            return json.loads(raw)

        except Exception as e:
            if "RateLimit" in type(e).__name__:
                wait = RETRY_BASE_SEC * (2 ** attempt) + random.uniform(0, 1)
                print(f"    ⚠  Rate limit — waiting {wait:.1f}s (attempt {attempt+1})")
                time.sleep(wait)
            elif "json" in str(e).lower():
                return {"_parse_error": str(e)}
            else:
                wait = RETRY_BASE_SEC * (2 ** attempt)
                print(f"    ⚠  API error: {e} — waiting {wait:.1f}s")
                time.sleep(wait)

    return {"_parse_error": f"Max retries ({MAX_RETRIES}) exceeded"}


def run_tier2(convs: dict, model: str, limit: Optional[int],
              force: bool) -> list[T2Result]:
    """Run Tier 2 LLM judge on all (or limited) conversations."""
    if not _HAS_ANTHROPIC:
        print("  ✗ anthropic package not installed — skipping Tier 2")
        return []

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("  ✗ ANTHROPIC_API_KEY not set — skipping Tier 2")
        return []

    client = anthropic.Anthropic(api_key=api_key)
    CACHE_DIR.mkdir(exist_ok=True)

    print(f"\n[TIER 2] Running LLM-as-judge ({model}) …")

    results = []
    keys = list(convs.keys())
    if limit:
        keys = keys[:limit]
    total = len(keys)

    for i, (variant, gen_idx) in enumerate(keys, 1):
        conv = convs[(variant, gen_idx)]
        size   = variant.split("_", 1)[1] if "_" in variant else variant
        is_lora = variant.startswith("lora_")

        # Skip if no predicted or reference turns
        if not conv.predicted_turns or not conv.reference_turns:
            print(f"  → [{i}/{total}] {variant} / gen_idx={gen_idx}: skipped (empty turns)")
            continue

        cached_data = None if force else load_cache_t2(variant, gen_idx)
        is_cached   = cached_data is not None

        if not is_cached:
            print(f"  → [{i}/{total}] {variant} / gen_idx={gen_idx}: calling judge …")
            judge_response = call_judge_t2(client, conv, model)
            save_cache_t2(variant, gen_idx, judge_response)
            time.sleep(CALL_DELAY_SEC)
        else:
            print(f"  → [{i}/{total}] {variant} / gen_idx={gen_idx}: loaded from cache")
            judge_response = cached_data

        ctype = conv.conversation_type

        r = T2Result(
            model_variant   = variant,
            model_size      = size,
            is_lora         = is_lora,
            generation_idx  = gen_idx,
            knowledge_level = conv.knowledge_level,
            emotional_state = conv.emotional_state,
            communication_style = conv.communication_style,
            goal_clarity    = conv.goal_clarity,
            certainty       = conv.certainty,
            conversation_type = ctype,
            model_used      = model,
            cached          = is_cached,
        )

        if "_parse_error" in judge_response:
            r.parse_error = judge_response["_parse_error"]
            results.append(r)
            continue

        def _score(key: str) -> float:
            try:
                return max(1.0, min(5.0, float(judge_response.get(key, 0))))
            except (TypeError, ValueError):
                return 0.0

        # ── Extract scores by conversation type ───────────────────────────────
        r.uncertainty_score = _score("uncertainty_score")
        r.pragmatic_score   = _score("pragmatic_score")
        r.persona_score     = _score("persona_score")
        r.uncertainty_rationale = judge_response.get("uncertainty_rationale", "")
        r.pragmatic_rationale   = judge_response.get("pragmatic_rationale", "")
        r.persona_rationale     = judge_response.get("persona_rationale", "")

        if ctype == "type_a":
            r.depth_first_score         = _score("depth_first_score")
            r.prior_belief_score        = _score("prior_belief_score")
            r.info_incompleteness_score = _score("info_incompleteness_score")
            r.depth_first_rationale     = judge_response.get("depth_first_rationale", "")
            r.prior_belief_rationale    = judge_response.get("prior_belief_rationale", "")
            # overall: depth_first(0.20) + uncertainty(0.15) + prior_belief(0.30)
            #          + info_incompleteness(0.20) + persona(0.15)
            r.overall_score = round(
                r.depth_first_score         * 0.20 +
                r.uncertainty_score         * 0.15 +
                r.prior_belief_score        * 0.30 +
                r.info_incompleteness_score * 0.20 +
                r.persona_score             * 0.15,
                3,
            )

        elif ctype == "type_b":
            r.error_detection_score          = _score("error_detection_score")
            r.pushback_calibration_score     = _score("pushback_calibration_score")
            r.error_detection_rationale      = judge_response.get("error_detection_rationale", "")
            r.pushback_calibration_rationale = judge_response.get("pushback_calibration_rationale", "")
            # overall: error_detection(0.35) + uncertainty(0.15) + pushback_calibration(0.25)
            #          + pragmatic(0.10) + persona(0.15)
            r.overall_score = round(
                r.error_detection_score      * 0.35 +
                r.uncertainty_score          * 0.15 +
                r.pushback_calibration_score * 0.25 +
                r.pragmatic_score            * 0.10 +
                r.persona_score              * 0.15,
                3,
            )

        else:  # success
            r.depth_first_score = _score("depth_first_score")
            r.info_drip_score   = _score("info_drip_score")
            r.depth_first_rationale = judge_response.get("depth_first_rationale", "")
            r.info_drip_rationale   = judge_response.get("info_drip_rationale", "")
            r.overall_score = round(
                r.depth_first_score * DIMENSION_WEIGHTS["depth_first"] +
                r.uncertainty_score * DIMENSION_WEIGHTS["uncertainty"] +
                r.info_drip_score   * DIMENSION_WEIGHTS["info_drip"]   +
                r.pragmatic_score   * DIMENSION_WEIGHTS["pragmatic"]   +
                r.persona_score     * DIMENSION_WEIGHTS["persona"],
                3,
            )

        r.standout_divergence = judge_response.get("standout_divergence", "")
        r.standout_match      = judge_response.get("standout_match", "")

        results.append(r)

    print(f"  ✓ Tier 2 complete: {len(results)} results")
    return results


def save_t2_csv(results: list[T2Result], path: Path):
    if not results:
        return
    rows = [asdict(r) for r in results]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✓ {path.name} written ({len(rows)} rows)")

# ─────────────────────────────────────────────────────────────────────────────
#  Aggregation helpers
# ─────────────────────────────────────────────────────────────────────────────

def group_by_variant(results: list) -> dict[str, list]:
    out: dict[str, list] = defaultdict(list)
    for r in results:
        out[r.model_variant].append(r)
    return dict(out)


def avg(items: list, attr: str) -> float:
    vals = [getattr(r, attr) for r in items
            if getattr(r, attr, None) not in (None, 0.0) or attr.startswith("delta")]
    numeric = []
    for v in vals:
        try:
            numeric.append(float(v))
        except (TypeError, ValueError):
            pass
    return safe_mean(numeric)

# ─────────────────────────────────────────────────────────────────────────────
#  Terminal summary
# ─────────────────────────────────────────────────────────────────────────────

def print_t1_summary(results: list[T1Result]):
    print("\n" + "═" * 80)
    print("TIER 1 — CONVERSATION FIDELITY SUMMARY")
    print("═" * 80)

    by_variant = group_by_variant(results)
    variants = [v for v in ALL_VARIANTS if v in by_variant]

    # Table 1: Length & lexical deltas
    print("\n── A/B. Length & Lexical Deltas (pred − ref) ──")
    header = f"{'Model':<14} {'ΔWords/turn':>12} {'ΔHedge/turn':>12} {'ΔCertainty/turn':>16} {'ΔPromise/turn':>14}"
    print(header)
    print("─" * len(header))
    for v in variants:
        rs = by_variant[v]
        print(
            f"{v:<14} "
            f"{avg(rs, 'delta_words_mean'):>+12.3f} "
            f"{avg(rs, 'delta_hedge_rate'):>+12.3f} "
            f"{avg(rs, 'delta_certainty_rate'):>+16.3f} "
            f"{avg(rs, 'delta_promise_rate'):>+14.3f}"
        )

    # Table 2: Info density & structural flags
    print("\n── C/D. Info Density & Breadth-First ──")
    header2 = f"{'Model':<14} {'Pred InfoDens':>14} {'Ref InfoDens':>13} {'ΔInfoDens':>10} {'BF pred %':>10} {'BF ref %':>9}"
    print(header2)
    print("─" * len(header2))
    for v in variants:
        rs = by_variant[v]
        n = len(rs)
        pred_bf = sum(1 for r in rs if r.pred_breadth_first) / n * 100 if n else 0
        ref_bf  = sum(1 for r in rs if r.ref_breadth_first)  / n * 100 if n else 0
        print(
            f"{v:<14} "
            f"{avg(rs, 'pred_info_density'):>14.3f} "
            f"{avg(rs, 'ref_info_density'):>13.3f} "
            f"{avg(rs, 'delta_info_density'):>+10.3f} "
            f"{pred_bf:>9.1f}% "
            f"{ref_bf:>8.1f}%"
        )

    # Table 3: Fidelity flags
    print("\n── Fidelity Flags ──")
    flag_cols = [
        ("flag_length_inflated",   "LenInflated"),
        ("flag_length_deflated",   "LenDeflated"),
        ("flag_over_certain",      "OverCertain"),
        ("flag_under_hedge",       "UnderHedge"),
        ("flag_info_front_loaded", "FrontLoad"),
        ("flag_breadth_first_new", "BreadthNew"),
        ("flag_role_confused",     "RoleConf"),
    ]
    col_width = 12
    header3 = f"{'Model':<14}" + "".join(f"{c[1]:>{col_width}}" for c in flag_cols)
    print(header3)
    print("─" * len(header3))
    for v in variants:
        rs = by_variant[v]
        n = len(rs)
        row = f"{v:<14}"
        for attr, _ in flag_cols:
            rate = sum(1 for r in rs if getattr(r, attr, False)) / n * 100 if n else 0
            row += f"{rate:>{col_width}.1f}%"
        print(row)

    total_flags = sum(r.fidelity_flag_count for r in results)
    print(f"\n  Total fidelity flags raised: {total_flags} across {len(results)} conversations")


def print_t2_summary(results: list[T2Result]):
    if not results:
        return

    print("\n" + "═" * 80)
    print("TIER 2 — LLM JUDGE FIDELITY SUMMARY")
    print("═" * 80)

    by_variant = group_by_variant(results)
    variants = [v for v in ALL_VARIANTS if v in by_variant]

    dims = ["depth_first_score", "uncertainty_score", "info_drip_score",
            "pragmatic_score", "persona_score", "overall_score"]
    dim_labels = ["DepthFirst", "Uncertainty", "InfoDrip", "Pragmatic", "Persona", "OVERALL"]

    col_w = 12
    header = f"{'Model':<14}" + "".join(f"{d:>{col_w}}" for d in dim_labels)
    print("\n" + header)
    print("─" * len(header))
    for v in variants:
        rs = [r for r in by_variant[v] if not r.parse_error]
        if not rs:
            continue
        row = f"{v:<14}"
        for attr in dims:
            row += f"{avg(rs, attr):>{col_w}.2f}"
        print(row)

    # Weakest dimension across LoRA variants
    lora_results = [r for r in results if r.is_lora and not r.parse_error]
    if lora_results:
        dim_avgs = {d: avg(lora_results, d) for d in dims[:-1]}
        weakest  = min(dim_avgs, key=dim_avgs.get)
        strongest = max(dim_avgs, key=dim_avgs.get)
        wl = weakest.replace("_score", "").replace("_", "-")
        sl = strongest.replace("_score", "").replace("_", "-")
        print(f"\n  LoRA weakest dimension:   {wl} ({dim_avgs[weakest]:.2f})")
        print(f"  LoRA strongest dimension: {sl} ({dim_avgs[strongest]:.2f})")

# ─────────────────────────────────────────────────────────────────────────────
#  Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def fig_t1_overview(results: list[T1Result]):
    """Figure cp_01: Predicted vs reference length + hedge rate per variant."""
    _set_style()
    by_variant = group_by_variant(results)
    variants = [v for v in ALL_VARIANTS if v in by_variant]
    x = np.arange(len(variants))
    w = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Tier 1: Predicted vs Reference Conversation Properties", fontweight="bold")

    metrics = [
        ("pred_words_mean", "ref_words_mean", "Mean words/turn"),
        ("pred_hedge_rate", "ref_hedge_rate", "Hedge rate"),
        ("pred_certainty_rate", "ref_certainty_rate", "Certainty rate"),
    ]

    for ax, (pred_col, ref_col, title) in zip(axes, metrics):
        pred_vals = [avg(by_variant[v], pred_col) for v in variants]
        ref_vals  = [avg(by_variant[v], ref_col)  for v in variants]

        bars_ref  = ax.bar(x - w/2, ref_vals,  w, label="Reference",
                           color=[PALETTE.get("reference", "#1B3A4B")] * len(variants), alpha=0.85)
        bars_pred = ax.bar(x + w/2, pred_vals, w, label="Predicted",
                           color=[PALETTE.get(v, "#ADB5BD") for v in variants], alpha=0.9)

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=35, ha="right")
        ax.legend(fontsize=9)

    plt.tight_layout()
    out = IMAGES_DIR / "cp_01_t1_overview.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


def fig_t1_deltas(results: list[T1Result]):
    """Figure cp_02: Delta bar charts (pred − ref) for key metrics."""
    _set_style()
    by_variant = group_by_variant(results)
    variants = [v for v in ALL_VARIANTS if v in by_variant]
    x = np.arange(len(variants))

    delta_cols = [
        ("delta_words_mean",    "Δ Mean words/turn"),
        ("delta_hedge_rate",    "Δ Hedge rate"),
        ("delta_certainty_rate","Δ Certainty rate"),
        ("delta_info_density",  "Δ Info density (T1)"),
    ]

    fig, axes = plt.subplots(1, len(delta_cols), figsize=(16, 5))
    fig.suptitle("Tier 1: Predicted − Reference Deltas (lower |Δ| = higher fidelity)",
                 fontweight="bold")

    for ax, (col, title) in zip(axes, delta_cols):
        vals   = [avg(by_variant[v], col) for v in variants]
        colors = ["#E76F51" if v > 0 else "#4C9BE8" for v in vals]
        ax.bar(x, vals, color=colors, alpha=0.9)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=35, ha="right")

    plt.tight_layout()
    out = IMAGES_DIR / "cp_02_t1_deltas.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


def fig_t1_flags(results: list[T1Result]):
    """Figure cp_03: Fidelity flag rates per variant."""
    _set_style()
    by_variant = group_by_variant(results)
    variants = [v for v in ALL_VARIANTS if v in by_variant]

    flag_cols = [
        ("flag_length_inflated",   "Length\nInflated"),
        ("flag_length_deflated",   "Length\nDeflated"),
        ("flag_over_certain",      "Over\nCertain"),
        ("flag_under_hedge",       "Under\nHedge"),
        ("flag_info_front_loaded", "Front\nLoaded"),
        ("flag_breadth_first_new", "Breadth\nFirst"),
        ("flag_role_confused",     "Role\nConfusion"),
    ]

    n_flags = len(flag_cols)
    flag_matrix = np.zeros((len(variants), n_flags))
    for i, v in enumerate(variants):
        rs = by_variant[v]
        n  = len(rs)
        for j, (attr, _) in enumerate(flag_cols):
            flag_matrix[i, j] = sum(1 for r in rs if getattr(r, attr, False)) / n * 100 if n else 0

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Tier 1: Fidelity Flag Rates per Model Variant (%)", fontweight="bold")

    im = ax.imshow(flag_matrix, cmap="RdYlGn_r", vmin=0, vmax=80, aspect="auto")
    ax.set_xticks(range(n_flags))
    ax.set_xticklabels([c[1] for c in flag_cols], fontsize=9)
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels(variants)

    for i in range(len(variants)):
        for j in range(n_flags):
            ax.text(j, i, f"{flag_matrix[i, j]:.0f}%",
                    ha="center", va="center", fontsize=8,
                    color="white" if flag_matrix[i, j] > 40 else "black")

    plt.colorbar(im, ax=ax, label="Flag rate (%)")
    plt.tight_layout()
    out = IMAGES_DIR / "cp_03_t1_flags.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


def fig_t2_scores(results: list[T2Result]):
    """Figure cp_04: Tier 2 overall fidelity score per variant."""
    if not results:
        return
    _set_style()
    valid = [r for r in results if not r.parse_error and r.overall_score > 0]
    if not valid:
        return

    by_variant = group_by_variant(valid)
    variants   = [v for v in ALL_VARIANTS if v in by_variant]

    dims = ["depth_first_score", "uncertainty_score", "info_drip_score",
            "pragmatic_score", "persona_score"]
    dim_labels = ["Depth-First", "Uncertainty", "Info Drip", "Pragmatic", "Persona"]

    # Grouped bars
    x   = np.arange(len(variants))
    w   = 0.14
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle("Tier 2: LLM Judge Fidelity Scores per Dimension (1–5)", fontweight="bold")

    dim_colors = ["#457B9D", "#2A9D8F", "#E9C46A", "#A8DADC", "#F4A261"]
    for di, (col, label, color) in enumerate(zip(dims, dim_labels, dim_colors)):
        vals = [avg(by_variant[v], col) for v in variants]
        offset = (di - len(dims) / 2) * w + w / 2
        ax.bar(x + offset, vals, w, label=label, color=color, alpha=0.9)

    ax.axhline(3, color="black", linewidth=0.8, linestyle="--", alpha=0.5, label="Midpoint (3)")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=35, ha="right")
    ax.set_ylim(1, 5.5)
    ax.set_ylabel("Fidelity score (1–5)")
    ax.legend(fontsize=8, ncol=3)

    plt.tight_layout()
    out = IMAGES_DIR / "cp_04_t2_scores.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


def fig_t1_by_conversation_type(t1_results: list["T1Result"]):
    """
    Figure cp_06: Tier 1 fidelity flag rates and probe signals broken out by
    conversation type (success / type_a / type_b). For type_b shows pushback
    miss rate; for type_a shows prior-belief-missing rate.
    """
    _set_style()
    if not t1_results:
        return

    lora_results = [r for r in t1_results if r.is_lora]
    if not lora_results:
        lora_results = t1_results

    conv_types  = ["success", "type_a", "type_b"]
    type_colors = {"success": "#4C9BE8", "type_a": "#E9C46A", "type_b": "#E76F51"}
    type_labels = {"success": "Success", "type_a": "Type A\n(inadvertent)", "type_b": "Type B\n(adversarial)"}

    # Metrics: (attr, display_label, applicable_types or None for all)
    metrics = [
        ("fidelity_flag_count",        "Mean Flag Count",    None),
        ("flag_role_confused",         "Role Confusion %",   None),
        ("flag_missed_pushback",       "Missed Pushback %",  ["type_b"]),
        ("flag_wrong_belief_missing",  "Belief Missing %",   ["type_a"]),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 5))
    fig.suptitle(
        "Tier 1 Fidelity Flags by Conversation Type  (LoRA variants)",
        fontweight="bold",
    )

    variants = sorted(set(r.model_variant for r in lora_results))

    for ax, (attr, label, applicable) in zip(axes, metrics):
        x = np.arange(len(variants))
        w = 0.22
        for ci, ctype in enumerate(conv_types):
            if applicable and ctype not in applicable:
                # Shade to indicate not applicable
                offset = (ci - 1) * w
                ax.bar(x + offset, [0] * len(variants), w * 0.9,
                       color="#EEEEEE", alpha=0.5, label=ctype if ci == 0 else "")
                continue

            vals = []
            for vname in variants:
                subset = [r for r in lora_results
                          if r.model_variant == vname and r.conversation_type == ctype]
                if not subset:
                    vals.append(0.0)
                    continue
                raw = [float(getattr(r, attr)) for r in subset]
                vals.append(round(np.mean(raw) * (100 if attr != "fidelity_flag_count" else 1), 2))
            offset = (ci - 1) * w
            bars = ax.bar(x + offset, vals, w * 0.9,
                          label=type_labels[ctype], color=type_colors[ctype], alpha=0.85)
            for bar, val in zip(bars, vals):
                if val > 0.01:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.5,
                            f"{val:.1f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([v.replace("_", " ").upper() for v in variants],
                           fontsize=8, rotation=25, ha="right")
        suffix = "%" if attr != "fidelity_flag_count" else ""
        ax.set_title(f"{label}{suffix}", fontsize=10)
        if applicable:
            ax.set_facecolor("#FAFAFA")
            ax.text(0.5, 0.95, f"({applicable[0]} only)",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=8, color="#999")

    axes[0].legend(fontsize=8, title="Conv. type", loc="upper right")
    fig.tight_layout()
    out = IMAGES_DIR / "cp_06_t1_by_conv_type.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


def fig_t2_by_dimension(results: list[T2Result]):
    """Figure cp_05: T2 overall score by persona dimension breakdowns."""
    if not results:
        return
    valid = [r for r in results if not r.parse_error and r.overall_score > 0]
    if not valid:
        return
    _set_style()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Tier 2: Fidelity Score Breakdown by Persona Attributes (LoRA only)",
                 fontweight="bold")

    lora_valid = [r for r in valid if r.is_lora]
    breakdowns = [
        ("emotional_state",     "Emotional State"),
        ("knowledge_level",     "Knowledge Level"),
        ("communication_style", "Communication Style"),
    ]

    for ax, (attr, title) in zip(axes, breakdowns):
        vals_by_cat: dict[str, list] = defaultdict(list)
        for r in lora_valid:
            cat = getattr(r, attr, "unknown") or "unknown"
            vals_by_cat[cat].append(r.overall_score)

        cats  = sorted(vals_by_cat.keys())
        means = [safe_mean(vals_by_cat[c]) for c in cats]
        stds  = [safe_std(vals_by_cat[c]) for c in cats]

        ax.barh(cats, means, xerr=stds, capsize=4, color="#4C9BE8", alpha=0.85)
        ax.axvline(3, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlim(1, 5)
        ax.set_title(title)
        ax.set_xlabel("Fidelity score")

    plt.tight_layout()
    out = IMAGES_DIR / "cp_05_t2_by_dimension.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")

# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Two-tier conversation-level evaluation of UserLM predictions"
    )
    parser.add_argument("--judge",       action="store_true",
                        help="Run Tier 2 LLM-as-judge (requires ANTHROPIC_API_KEY)")
    parser.add_argument("--judge-model", default=JUDGE_MODEL,
                        help=f"Anthropic model for Tier 2 (default: {JUDGE_MODEL})")
    parser.add_argument("--judge-limit", type=int, default=None,
                        help="Limit Tier 2 to first N conversations (for testing)")
    parser.add_argument("--force",       action="store_true",
                        help="Ignore Tier 2 cache and re-evaluate all conversations")
    args = parser.parse_args()

    IMAGES_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("eval_conversation_predictions.py")
    print("=" * 70)

    # ── Step 1: Load detailed CSV ────────────────────────────────────────────
    print("\n[STEP 1] Loading prediction_metrics_detailed.csv …")
    if not DETAILED_CSV.exists():
        print(f"  ✗ {DETAILED_CSV.name} not found. Run eval_predictions.py first.")
        return
    rows = load_detailed_csv(DETAILED_CSV)
    print(f"  → {len(rows)} rows loaded")

    # ── Step 2: Reconstruct conversations ────────────────────────────────────
    print("\n[STEP 2] Reconstructing conversations …")
    convs = reconstruct_conversations(rows)

    # ── Step 3: Tier 1 ──────────────────────────────────────────────────────
    print("\n[STEP 3] Tier 1 — Rule-based evaluation …")
    t1_results = run_tier1(convs)
    save_t1_csv(t1_results, T1_CSV)

    # ── Step 4: Tier 1 visualisations ────────────────────────────────────────
    print("\n[STEP 4] Generating Tier 1 visualisations …")
    fig_t1_overview(t1_results)
    fig_t1_deltas(t1_results)
    fig_t1_flags(t1_results)
    fig_t1_by_conversation_type(t1_results)

    # ── Step 5: Tier 2 (optional) ────────────────────────────────────────────
    t2_results = []
    if args.judge:
        print("\n[STEP 5] Tier 2 — LLM-as-judge evaluation …")
        t2_results = run_tier2(
            convs, args.judge_model, args.judge_limit, args.force
        )
        save_t2_csv(t2_results, T2_CSV)
        fig_t2_scores(t2_results)
        fig_t2_by_dimension(t2_results)
    else:
        print("\n[STEP 5] Tier 2 skipped (pass --judge to enable)")

    # ── Step 6: Terminal summaries ────────────────────────────────────────────
    print("\n[STEP 6] Summaries …")
    print_t1_summary(t1_results)
    if t2_results:
        print_t2_summary(t2_results)

    print("\n" + "=" * 70)
    print("Done.")
    print(f"  Tier 1 CSV:  {T1_CSV.name}")
    if t2_results:
        print(f"  Tier 2 CSV:  {T2_CSV.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
