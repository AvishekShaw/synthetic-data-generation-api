#!/usr/bin/env python3
"""
eval_predictions.py
Multi-model prediction evaluation for UserLM fine-tuned Gemma variants.

Loads prediction JSONL files (one per model size), matches examples positionally,
computes new metrics on top of existing ones, and produces model-wise CSVs + graphs.

Key questions answered:
  a) Category-level: does the model learn terse/direct/indirect behaviour differently?
  b) Cross-model: 4B vs 12B vs 27B — where does scale help?
  c) EOS: does the LoRA model learn to predict "" to end conversations?

New metrics added on top of existing (loss, perplexity, BLEU, ROUGE):
  - normalized_exact_match  : exact match after stripping <eos> and lowercasing
  - is_eos_example          : whether expected output is just <eos>
  - eos_correct             : (EOS examples only) did model predict empty string?
  - eos_false_pos           : (non-EOS examples only) did model wrongly predict empty?
  - length_ratio_error      : |1 - predicted_tokens/expected_tokens|, lower = better
  - style_compliance        : rule-based check of predicted output vs persona style
  - meteor_score            : METEOR (better than BLEU for short/paraphrased text)

Outputs:
  - prediction_metrics_detailed.csv  — one row per (model_variant × example)
  - category_summary.csv             — aggregated metrics per (model × slice)
  - model_comparison.csv             — pairwise win rates + Wilcoxon p-values
  - images/13_category_heatmap.png
  - images/14_model_comparison.png
  - images/15_eos_analysis.png
  - images/16_efficiency_frontier.png
  - images/17_style_compliance.png
"""

import csv
import json
import math
import os
import re
import time
import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass, field
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
from scipy.stats import wilcoxon, spearmanr

import nltk
for _res in ("wordnet", "omw-1.4"):
    nltk.download(_res, quiet=True)
from nltk.translate.meteor_score import meteor_score as _nltk_meteor
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ── Optional heavy dependencies (graceful fallback) ─────────────────────
_HAS_BERTSCORE = False
try:
    from bert_score import score as _bert_score_fn
    _HAS_BERTSCORE = True
except ImportError:
    pass

_HAS_BLEURT = False
try:
    from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
    _HAS_BLEURT = True
except ImportError:
    pass

_HAS_ANTHROPIC = False
try:
    import anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
#  HARDCODED PATHS — update these three lines before running
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent

def _find_predictions(stem: str) -> Path:
    for ext in (".jsonl", ".json"):
        p = BASE_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    return BASE_DIR / f"{stem}.jsonl"   # fallback — will be flagged as missing

MODEL_FILES = {
    "gemma_4b":  _find_predictions("predictions_gemma_4b"),
    "gemma_12b": _find_predictions("predictions_gemma_12b"),
    "gemma_27b": _find_predictions("predictions_gemma_27b"),
}

# Display names and parameter counts (used in efficiency plots)
MODEL_INFO = {
    "gemma_4b":  {"display": "Gemma 4B",  "params_b": 4},
    "gemma_12b": {"display": "Gemma 12B", "params_b": 12},
    "gemma_27b": {"display": "Gemma 27B", "params_b": 27},
}

# ─────────────────────────────────────────────────────────────────────────────
#  Output paths
# ─────────────────────────────────────────────────────────────────────────────

IMAGES_DIR        = BASE_DIR / "images"
DETAILED_CSV      = BASE_DIR / "prediction_metrics_detailed.csv"
CATEGORY_CSV      = BASE_DIR / "category_summary.csv"
COMPARISON_CSV    = BASE_DIR / "model_comparison.csv"
JUDGE_CACHE_DIR   = BASE_DIR / "judge_cache"
CONVERSATION_CSV  = BASE_DIR / "conversation_metrics.csv"

# ── LLM-as-judge config ─────────────────────────────────────────────────
JUDGE_MODEL       = "claude-sonnet-4-6"

# ── BERTScore config ────────────────────────────────────────────────────
BERTSCORE_MODEL   = "microsoft/deberta-xlarge-mnli"

# ── BLEURT config ───────────────────────────────────────────────────────
BLEURT_CHECKPOINT = "lucadiliello/BLEURT-20"

# ─────────────────────────────────────────────────────────────────────────────
#  Visual style  (mirrors existing eval scripts)
# ─────────────────────────────────────────────────────────────────────────────

# One colour per model variant — base models are desaturated versions of LoRA colours
MODEL_COLORS = {
    "base_4b":  "#A8D5E2",
    "lora_4b":  "#2176AE",
    "base_12b": "#F4C095",
    "lora_12b": "#E76F51",
    "base_27b": "#B5D5C5",
    "lora_27b": "#2A9D8F",
}

PERSONA_COLORS = {
    # communication style
    "direct":   "#457B9D",
    "terse":    "#2A9D8F",
    "indirect": "#E9C46A",
    # emotional state
    "calm":              "#4C9BE8",
    "mildly_frustrated": "#F4A261",
    "escalating":        "#C1121F",
    # knowledge
    "novice":       "#BDE0FE",
    "intermediate": "#6A8EAE",
    "expert":       "#1B3A4B",
}

CATEGORY_SLICES = [
    ("communication_style", "persona"),
    ("emotional_state",     "persona"),
    ("knowledge_level",     "persona"),
    ("goal_clarity",        "persona"),
    ("certainty",           "scenario"),
    ("conversation_type",   "data_type"),   # success | type_a | type_b
]

# Terse word-count threshold (≤ this = compliant)
TERSE_MAX_WORDS    = 12
DIRECT_MAX_WORDS   = 20
INDIRECT_MIN_WORDS = 8   # indirect users tend to give more context


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
    })


# ─────────────────────────────────────────────────────────────────────────────
#  Normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Strip <eos>, lowercase, remove leading/trailing whitespace."""
    return text.replace("<eos>", "").strip().lower()


def word_count(text: str) -> int:
    return len(text.split()) if text.strip() else 0


# ─────────────────────────────────────────────────────────────────────────────
#  Style compliance  (rule-based, no LLM needed)
# ─────────────────────────────────────────────────────────────────────────────

HEDGE_PHRASES = [
    "not sure", "i think", "i believe", "maybe", "perhaps", "possibly",
    "probably", "might", "could be", "i guess", "i don't know", "idk",
    "honestly", "kind of", "sort of", "not certain",
]

FRUSTRATION_MARKERS = [
    "look", "already said", "already told", "come on", "seriously",
    "just want", "all i want", "that's why", "thats why",
    "i already", "again",
]

# Pushback phrases for type_b conversations (matches generator's list, extended)
PUSHBACK_PHRASES = [
    "that's not right", "that's incorrect", "are you sure", "i thought",
    "i was told", "i've read", "according to", "that doesn't sound right",
    "wait,", "hold on", "actually,", "i don't think that's", "you said earlier",
    "but earlier", "you just said", "that contradicts", "that can't be right",
    "that's wrong", "no, it's", "i believe it's", "reg e", "regulation e",
    "that's not what", "isn't it", "i'm pretty sure", "i read that",
]

# Per-scenario keywords that signal the prior belief is being expressed (type_a only).
# Keyed by scenario_idx % 8 (type_a uses the same 8 scenarios as success,
# offset by generation_idx 1000+).
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


def conversation_type_from_meta(meta: dict) -> str:
    """Derive conversation type from _meta dict."""
    pt = meta.get("probing_type", "")
    if pt == "inadvertent":
        return "type_a"
    if pt == "adversarial":
        return "type_b"
    if meta.get("goal_completed") is False:
        return "failure"
    return "success"


def scenario_idx_from_gen_idx(gen_idx, conv_type: str) -> int:
    """Extract scenario index (0–7) from generation_idx for prior belief lookup."""
    try:
        g = int(gen_idx or 0)
    except (TypeError, ValueError):
        return 0
    if conv_type == "type_b":
        return (g - 2000) % 8
    if conv_type == "type_a":
        return (g - 1000) % 8
    return g % 8


def check_pushback_turn(text: str) -> bool:
    """True if this turn contains pushback language (type_b signal)."""
    tl = text.lower()
    return any(p in tl for p in PUSHBACK_PHRASES)


def check_prior_belief_expressed(text: str, scenario_idx: int) -> bool:
    """True if this turn expresses the wrong prior belief for the scenario (type_a signal)."""
    tl = text.lower()
    keywords = PRIOR_BELIEF_KEYWORDS.get(scenario_idx % 8, [])
    return any(k in tl for k in keywords)


def style_compliance(predicted_norm: str, persona: dict,
                     conversation_type: str = "success") -> float:
    """
    Returns a 0.0–1.0 compliance score.
    For EOS predictions (empty string) returns 1.0 — no style to violate.
    """
    if not predicted_norm:
        return 1.0

    style   = persona.get("communication_style", "")
    emotion = persona.get("emotional_state", "")
    wc      = word_count(predicted_norm)
    checks, score = 0, 0.0

    # ── Communication style ──────────────────────────────────────────
    if style == "terse":
        checks += 1
        score  += 1.0 if wc <= TERSE_MAX_WORDS else max(0.0, 1 - (wc - TERSE_MAX_WORDS) / 15)

    elif style == "direct":
        checks += 1
        score  += 1.0 if wc <= DIRECT_MAX_WORDS else max(0.0, 1 - (wc - DIRECT_MAX_WORDS) / 20)

    elif style == "indirect":
        checks += 1
        has_hedge = any(p in predicted_norm for p in HEDGE_PHRASES)
        long_enough = wc >= INDIRECT_MIN_WORDS
        score += (0.5 * has_hedge) + (0.5 * long_enough)

    # ── Emotional state ──────────────────────────────────────────────
    # For type_b (adversarial probing), pushback / assertive language is correct
    # behaviour — even a normally "calm" persona should push back on agent errors.
    # Suppress the emotion check so we don't penalize legitimate pushback.
    if conversation_type != "type_b":
        if emotion == "mildly_frustrated":
            checks += 1
            has_frustration = any(m in predicted_norm for m in FRUSTRATION_MARKERS)
            score += 1.0 if has_frustration else 0.4   # partial credit (frustration subtle)
        elif emotion == "calm":
            checks += 1
            no_frustration = not any(m in predicted_norm for m in FRUSTRATION_MARKERS)
            score += 1.0 if no_frustration else 0.0

    return round(score / checks, 4) if checks > 0 else 1.0   # 1.0 = nothing to violate


# ─────────────────────────────────────────────────────────────────────────────
#  METEOR wrapper
# ─────────────────────────────────────────────────────────────────────────────

_nltk_ready = False

def compute_meteor(reference: str, hypothesis: str) -> float:
    global _nltk_ready
    if not _nltk_ready:
        print("  Downloading NLTK resources (first run only) …", flush=True)
        for _res in ("wordnet", "omw-1.4"):
            nltk.download(_res, quiet=True)
        _nltk_ready = True

    if not reference or not hypothesis:
        return 0.0
    try:
        return round(float(_nltk_meteor([reference.split()], hypothesis.split())), 4)
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Entity-level precision / recall  (banking domain)
# ─────────────────────────────────────────────────────────────────────────────

# Regex patterns for banking entities
_AMOUNT_RE   = re.compile(r"\$?\d+(?:\.\d{1,2})?")
_DATE_RE     = re.compile(
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?",
    re.IGNORECASE,
)
_CARD_RE     = re.compile(r"\d{4}(?:\s*[\*x]+\s*){0,3}\d{4}")
_MERCHANT_NAMES = [
    "netflix", "amazon", "spotify", "apple", "google", "uber", "lyft",
    "paypal", "venmo", "zelle", "walmart", "target", "costco", "chase",
    "wells fargo", "bank of america", "citi", "capital one",
]


def extract_entities(text: str) -> set:
    """Extract normalised banking entities from text."""
    text_lower = text.lower()
    entities = set()
    for m in _AMOUNT_RE.finditer(text_lower):
        # Normalise: strip $ and trailing .00
        val = m.group().replace("$", "").strip()
        entities.add(f"AMT:{val}")
    for m in _DATE_RE.finditer(text_lower):
        entities.add(f"DATE:{m.group().strip()}")
    for m in _CARD_RE.finditer(text):
        digits = re.sub(r"[^\d]", "", m.group())
        if len(digits) >= 4:
            entities.add(f"CARD:{digits[-4:]}")
    for merchant in _MERCHANT_NAMES:
        if merchant in text_lower:
            entities.add(f"MERCHANT:{merchant}")
    return entities


def entity_precision_recall(reference: str, predicted: str) -> dict:
    """Compute entity-level precision, recall, F1 between reference and predicted."""
    ref_ents  = extract_entities(reference)
    pred_ents = extract_entities(predicted)
    if not ref_ents and not pred_ents:
        return {"entity_precision": 1.0, "entity_recall": 1.0, "entity_f1": 1.0,
                "entity_count_ref": 0, "entity_count_pred": 0}
    if not pred_ents:
        return {"entity_precision": 1.0, "entity_recall": 0.0, "entity_f1": 0.0,
                "entity_count_ref": len(ref_ents), "entity_count_pred": 0}
    if not ref_ents:
        return {"entity_precision": 0.0, "entity_recall": 1.0, "entity_f1": 0.0,
                "entity_count_ref": 0, "entity_count_pred": len(pred_ents)}
    tp = len(ref_ents & pred_ents)
    p  = tp / len(pred_ents) if pred_ents else 0.0
    r  = tp / len(ref_ents)  if ref_ents  else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"entity_precision": round(p, 4), "entity_recall": round(r, 4),
            "entity_f1": round(f1, 4),
            "entity_count_ref": len(ref_ents), "entity_count_pred": len(pred_ents)}


# ─────────────────────────────────────────────────────────────────────────────
#  Lexical diversity — Type-Token Ratio
# ─────────────────────────────────────────────────────────────────────────────

def type_token_ratio(text: str) -> float:
    """Compute type-token ratio. Returns 0.0 for empty text."""
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    return round(len(set(tokens)) / len(tokens), 4)


# ─────────────────────────────────────────────────────────────────────────────
#  Role confusion detector
# ─────────────────────────────────────────────────────────────────────────────

ASSISTANT_LANGUAGE = [
    "i'd be happy to", "i can help", "let me help", "i can assist",
    "how can i help", "is there anything else", "i'll look into",
    "thank you for contacting", "let me check", "i understand your concern",
    "i apologize for", "we appreciate your patience",
]

AGENT_PREFIX_RE = re.compile(r"^\s*agent\s*:", re.IGNORECASE)


def detect_role_confusion(predicted_norm: str) -> dict:
    """
    Detect if the predicted user turn contains assistant-side language.
    Returns a dict with:
      - role_confused: bool
      - role_confusion_signals: list of matched patterns
    """
    signals = []
    if AGENT_PREFIX_RE.match(predicted_norm):
        signals.append("agent_prefix")
    for phrase in ASSISTANT_LANGUAGE:
        if phrase in predicted_norm:
            signals.append(phrase)
    return {
        "role_confused": len(signals) > 0,
        "role_confusion_signals": signals[:3],  # keep top 3 for CSV readability
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Hedging calibration
# ─────────────────────────────────────────────────────────────────────────────

# Promise language (LLM-like over-commitment — Wang et al. 2025)
PROMISE_PHRASES = [
    "i will", "i'll", "i can", "i am going to", "i'm going to",
    "definitely", "absolutely", "certainly", "of course", "no problem",
    "right away", "immediately",
]


def compute_hedge_rate(text: str) -> float:
    """Fraction of hedge phrases found in text (0–1)."""
    text_lower = text.lower()
    if not text_lower.strip():
        return 0.0
    hits = sum(1 for p in HEDGE_PHRASES if p in text_lower)
    return round(hits / len(HEDGE_PHRASES), 4)


def compute_promise_rate(text: str) -> float:
    """Fraction of promise/commitment phrases found in text (0–1)."""
    text_lower = text.lower()
    if not text_lower.strip():
        return 0.0
    hits = sum(1 for p in PROMISE_PHRASES if p in text_lower)
    return round(hits / len(PROMISE_PHRASES), 4)


# ─────────────────────────────────────────────────────────────────────────────
#  Batch BERTScore (runs after all examples collected)
# ─────────────────────────────────────────────────────────────────────────────

def compute_bertscore_batch(
    references: list[str],
    predictions: list[str],
    model_type: str = BERTSCORE_MODEL,
) -> tuple[list[float], list[float], list[float]]:
    """
    Compute BERTScore for all (reference, prediction) pairs in a single batch.
    Returns (precisions, recalls, f1s) as lists of floats.
    Falls back to [None]*n if bert_score is not installed.
    """
    n = len(references)
    if not _HAS_BERTSCORE:
        print("    ⚠  bert-score not installed — skipping BERTScore. "
              "Install with: pip install bert-score")
        return [None] * n, [None] * n, [None] * n

    # Replace empty strings with a placeholder (BERTScore errors on empty)
    refs  = [r if r.strip() else "[empty]" for r in references]
    preds = [p if p.strip() else "[empty]" for p in predictions]

    print(f"    Computing BERTScore for {n} pairs (model={model_type}) …")
    P, R, F1 = _bert_score_fn(
        preds, refs,
        lang="en",
        model_type=model_type,
        rescale_with_baseline=True,
        verbose=False,
    )
    return (
        [round(x.item(), 4) for x in P],
        [round(x.item(), 4) for x in R],
        [round(x.item(), 4) for x in F1],
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Batch BLEURT (runs after all examples collected)
# ─────────────────────────────────────────────────────────────────────────────

_bleurt_model = None
_bleurt_tokenizer = None


def compute_bleurt_batch(
    references: list[str],
    predictions: list[str],
    batch_size: int = 32,
) -> list[float]:
    """
    Compute BLEURT scores for all pairs. Returns list of floats.
    Falls back to [None]*n if bleurt-pytorch is not installed.
    """
    global _bleurt_model, _bleurt_tokenizer
    n = len(references)
    if not _HAS_BLEURT:
        print("    ⚠  bleurt-pytorch not installed — skipping BLEURT. "
              "Install with: pip install bleurt-pytorch")
        return [None] * n

    import torch

    if _bleurt_model is None:
        print(f"    Loading BLEURT model ({BLEURT_CHECKPOINT}) …")
        _bleurt_tokenizer = BleurtTokenizer.from_pretrained(BLEURT_CHECKPOINT)
        _bleurt_model = BleurtForSequenceClassification.from_pretrained(BLEURT_CHECKPOINT)
        _bleurt_model.eval()

    refs  = [r if r.strip() else "[empty]" for r in references]
    preds = [p if p.strip() else "[empty]" for p in predictions]

    print(f"    Computing BLEURT for {n} pairs …")
    scores = []
    for i in range(0, n, batch_size):
        batch_refs  = refs[i:i + batch_size]
        batch_preds = preds[i:i + batch_size]
        inputs = _bleurt_tokenizer(
            batch_refs, batch_preds,
            padding=True, truncation=True, max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = _bleurt_model(**inputs)
        scores.extend([round(s.item(), 4) for s in out.logits.squeeze(-1)])
    return scores


# ─────────────────────────────────────────────────────────────────────────────
#  LLM-as-judge  (6-dimension turn-level rubric)
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_RUBRIC = """You are evaluating whether a candidate user turn matches the quality of a reference user turn in a banking fraud dispute conversation.

## Persona
- Communication style: {communication_style}
- Emotional state: {emotional_state}
- Knowledge level: {knowledge_level}
- Goal clarity: {goal_clarity}

## Conversation history
{conversation_history}

## Reference user turn (ground truth)
{expected_output}

## Candidate user turn (model prediction)
{predicted_output}

Score the candidate on each dimension below (1–5). Provide a one-sentence justification for each score, then the integer score.

### Dimension 1: Semantic Fidelity (1–5)
Does the candidate turn convey the same core meaning and intent as the reference?
5 = Identical meaning, possibly rephrased
4 = Same intent, minor information difference
3 = Same general topic but misses or adds a significant detail
2 = Related but different intent
1 = Unrelated or contradictory

### Dimension 2: Persona Voice Match (1–5)
Does the candidate sound like it was written by the described persona?
5 = Matches communication style, knowledge level, and emotional state perfectly
4 = Matches 2 of 3 persona dimensions, minor deviation on the third
3 = Generally appropriate but noticeably off on one dimension
2 = Clearly mismatched on style or emotion
1 = Sounds like a different person entirely, or sounds like an AI assistant

### Dimension 3: Conversational Coherence (1–5)
Given the conversation history, is this a natural, plausible next turn?
5 = Perfectly follows from the prior agent turn, advances the conversation
4 = Coherent response but slightly awkward transition
3 = Responsive to the general topic but ignores something the agent said
2 = Non-sequitur or repeats information already provided
1 = Incoherent or contradicts prior turns

### Dimension 4: Goal Directedness (1–5)
Does this turn move the conversation toward the persona's stated goal?
5 = Directly advances the goal
4 = Mostly advances the goal with minor tangent
3 = Neutral — doesn't help or hurt goal progress
2 = Stalls the conversation or introduces unnecessary complexity
1 = Actively counterproductive

### Dimension 5: Human Realism (1–5)
Does this read like a real human wrote it in a chat interface?
5 = Natural, appropriate informality, realistic information density
4 = Mostly natural with one slightly off element
3 = Functional but reads like a carefully constructed response
2 = Detectably LLM-generated — too polished, too comprehensive, too cooperative
1 = Obviously AI — uses assistant-side language, perfect grammar, bullet points

### Dimension 6: Information Calibration (1–5)
Does the candidate reveal the right amount of information for this persona and turn position?
5 = Information density matches what this persona would plausibly share at this point
4 = Slightly more or less information than expected, within normal variation
3 = Noticeably too much or too little info
2 = Significant mismatch — e.g., a novice perfectly articulating their dispute category
1 = Completely implausible information behavior

Return ONLY valid JSON (no markdown, no commentary outside the JSON):
{{
  "semantic_fidelity": {{"justification": "...", "score": N}},
  "persona_voice": {{"justification": "...", "score": N}},
  "conversational_coherence": {{"justification": "...", "score": N}},
  "goal_directedness": {{"justification": "...", "score": N}},
  "human_realism": {{"justification": "...", "score": N}},
  "information_calibration": {{"justification": "...", "score": N}}
}}"""

JUDGE_DIMENSIONS = [
    "semantic_fidelity", "persona_voice", "conversational_coherence",
    "goal_directedness", "human_realism", "information_calibration",
]


def _extract_conversation_history(input_text: str) -> str:
    """Pull conversation history from the input prompt."""
    marker = "Conversation so far:"
    idx = input_text.find(marker)
    if idx == -1:
        return "(First turn — no prior conversation)"
    rest = input_text[idx + len(marker):]
    # Trim the trailing instruction
    for end_marker in ["If your goal is complete", "Generate"]:
        eidx = rest.find(end_marker)
        if eidx != -1:
            rest = rest[:eidx]
    return rest.strip()


def run_llm_judge(
    example: dict,
    predicted: str,
    expected: str,
    persona: dict,
    model_variant: str,
    example_idx: int,
) -> dict:
    """
    Call Claude to score a single (predicted, expected) pair.
    Returns dict with scores for each dimension, or None-filled dict on failure.
    Results are cached to JUDGE_CACHE_DIR.
    """
    empty_result = {d: None for d in JUDGE_DIMENSIONS}
    empty_result["judge_justifications"] = {}

    if not _HAS_ANTHROPIC:
        return empty_result

    JUDGE_CACHE_DIR.mkdir(exist_ok=True)
    cache_file = JUDGE_CACHE_DIR / f"{model_variant}_{example_idx}.json"
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            return cached
        except Exception:
            pass  # re-evaluate on corrupt cache

    input_text = example.get("input", "")
    conv_history = _extract_conversation_history(input_text)

    prompt = JUDGE_RUBRIC.format(
        communication_style=persona.get("communication_style", "unknown"),
        emotional_state=persona.get("emotional_state", "unknown"),
        knowledge_level=persona.get("knowledge_level", "unknown"),
        goal_clarity=persona.get("goal_clarity", "unknown"),
        conversation_history=conv_history,
        expected_output=expected,
        predicted_output=predicted if predicted.strip() else "(empty — model predicted EOS)",
    )

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)

        result = {}
        justifications = {}
        for dim in JUDGE_DIMENSIONS:
            if dim in parsed and isinstance(parsed[dim], dict):
                result[dim] = int(parsed[dim].get("score", 0))
                justifications[dim] = parsed[dim].get("justification", "")
            else:
                result[dim] = None
        result["judge_justifications"] = justifications

        cache_file.write_text(json.dumps(result, indent=2))
        return result

    except Exception as e:
        print(f"    ⚠  Judge call failed for {model_variant}#{example_idx}: {e}")
        return empty_result


# ─────────────────────────────────────────────────────────────────────────────
#  Self-BLEU mode collapse detection (conversation-level, computed post-hoc)
# ─────────────────────────────────────────────────────────────────────────────

_smooth_fn = SmoothingFunction().method1

def compute_self_bleu(texts: list[str]) -> float:
    """
    Average pairwise BLEU across a set of texts.
    High self-BLEU = low diversity = mode collapse.
    """
    if len(texts) < 2:
        return 0.0
    tokenized = [t.lower().split() for t in texts if t.strip()]
    if len(tokenized) < 2:
        return 0.0
    scores = []
    for i, hyp in enumerate(tokenized):
        refs = [tokenized[j] for j in range(len(tokenized)) if j != i]
        try:
            s = sentence_bleu(refs, hyp, smoothing_function=_smooth_fn)
            scores.append(s)
        except Exception:
            pass
    return round(np.mean(scores), 4) if scores else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_file(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def verify_alignment(files_data: dict) -> int:
    """
    Assert that all files have the same number of examples and that
    the generation_idx at each position matches.
    Returns the number of examples.
    """
    sizes = {name: len(rows) for name, rows in files_data.items()}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"File lengths differ: {sizes}")

    n = list(sizes.values())[0]

    # Spot-check generation_idx alignment
    names = list(files_data.keys())
    for i in range(n):
        gen_ids = {
            name: files_data[name][i].get("_meta", {}).get("generation_idx")
            for name in names
        }
        unique_ids = set(gen_ids.values())
        if len(unique_ids) != 1:
            raise ValueError(
                f"generation_idx mismatch at position {i}: {gen_ids}. "
                "Files are not aligned — positional matching is unsafe."
            )

    print(f"  ✓  Alignment verified: {n} examples, generation_idx matches at all positions.")
    return n


# ─────────────────────────────────────────────────────────────────────────────
#  Per-example metric computation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExampleMetrics:
    # Identity
    example_idx:    int    = 0
    model_size:     str    = ""   # "gemma_4b" / "gemma_12b" / "gemma_27b"
    model_variant:  str    = ""   # "base_4b" / "lora_4b" etc.
    is_lora:        bool   = False

    # Metadata slices
    communication_style: str = ""
    emotional_state:     str = ""
    knowledge_level:     str = ""
    goal_clarity:        str = ""
    certainty:           str = ""
    information_completeness: str = ""
    expected_resolution: str = ""
    merchant_name:       str = ""
    amount:              str = ""
    transaction_date:    str = ""

    # Turn type
    is_eos_example: bool = False
    is_first_turn:  bool = False

    # Existing metrics (read from JSON)
    loss:                   float = 0.0
    perplexity:             float = 0.0
    avg_token_confidence:   float = 0.0
    bleu_score:             float = 0.0
    rouge_l_f1:             float = 0.0
    rouge_l_precision:      float = 0.0
    rouge_l_recall:         float = 0.0
    exact_match:            bool  = False
    token_ratio:            float = 0.0
    length_difference:      int   = 0
    predicted_tokens:       int   = 0
    expected_tokens:        int   = 0
    generation_time_sec:    float = 0.0

    # New metrics (v1 — existing)
    normalized_exact_match: bool  = False
    eos_correct:            object = None   # bool for EOS examples, None otherwise
    eos_false_pos:          object = None   # bool for non-EOS examples, None otherwise
    length_ratio_error:     float  = 0.0
    style_compliance_score: float  = 0.0
    meteor_score:           float  = 0.0

    # New metrics (v2 — semantic & entity)
    bertscore_f1:          object = None    # float or None if unavailable
    bertscore_precision:   object = None
    bertscore_recall:      object = None
    bleurt_score:          object = None    # float or None if unavailable
    entity_precision:      float  = 1.0
    entity_recall:         float  = 1.0
    entity_f1:             float  = 1.0
    entity_count_ref:      int    = 0
    entity_count_pred:     int    = 0

    # New metrics (v2 — behavioural signals)
    ttr_predicted:         float  = 0.0    # type-token ratio of predicted turn
    ttr_reference:         float  = 0.0    # type-token ratio of reference turn
    ttr_delta:             float  = 0.0    # |predicted - reference|
    hedge_rate_predicted:  float  = 0.0
    hedge_rate_reference:  float  = 0.0
    hedge_rate_delta:      float  = 0.0    # |predicted - reference|
    promise_rate_predicted: float = 0.0
    promise_rate_reference: float = 0.0
    promise_rate_delta:    float  = 0.0
    role_confused:         bool   = False
    role_confusion_signals: str   = ""     # comma-joined top signals

    # New metrics (v2 — LLM-as-judge, 1–5 scores)
    judge_semantic_fidelity:       object = None
    judge_persona_voice:           object = None
    judge_conversational_coherence: object = None
    judge_goal_directedness:       object = None
    judge_human_realism:           object = None
    judge_information_calibration: object = None
    judge_mean:                    object = None  # mean of 6 dimensions

    # Conversation-level identity (for grouping)
    generation_idx:        object = None

    # ── Conversation type and probe metadata ──────────────────────────────
    conversation_type:      str    = "success"  # success | failure | type_a | type_b
    wrong_prior_belief:     str    = ""          # type_a: the prior belief text
    agent_failure_mode:     str    = ""          # type_b: planted error category
    user_caught_error:      object = None        # type_b: bool — did user catch it?
    goal_completed:         object = None        # bool — did conversation reach goal?
    failure_mode:           str    = ""          # failure: dropout mode (impatience, loop_exit, …)
    # Turn-level probe signals
    is_pushback_turn:       bool   = False       # type_b: turn contains pushback language
    prior_belief_expressed: bool   = False       # type_a: turn expresses wrong prior belief

    # Raw text (for debugging)
    predicted_output:  str = ""
    expected_output:   str = ""


def compute_metrics_for_example(
    example: dict,
    model_key: str,      # "base_model" or "lora_model"
    model_size: str,     # "gemma_4b" etc.
    example_idx: int,
) -> ExampleMetrics:

    meta     = example.get("_meta", {})
    persona  = meta.get("persona", {})
    scenario = meta.get("scenario", {})
    inp      = example.get("input", "")
    expected = example.get("expected_output", "")

    # ── Probe / conversation-type metadata ───────────────────────────────
    conv_type        = conversation_type_from_meta(meta)
    gen_idx_val      = meta.get("generation_idx")
    sc_idx           = scenario_idx_from_gen_idx(gen_idx_val, conv_type)
    wrong_prior      = meta.get("wrong_prior_belief", "")
    agent_fail_mode  = meta.get("agent_failure_mode", "")
    caught_error     = meta.get("user_caught_error", None)
    goal_done        = meta.get("goal_completed", None)
    failure_mode     = meta.get("failure_mode", "")

    model_data = example.get(model_key, {})
    raw_metrics = model_data.get("metrics", {})
    predicted  = model_data.get("predicted_output", "")

    is_lora  = model_key == "lora_model"
    variant  = f"lora_{model_size.split('_')[1]}" if is_lora else f"base_{model_size.split('_')[1]}"

    norm_exp  = normalize(expected)
    norm_pred = normalize(predicted)

    # Explicit EOS check: expected_output is exactly "<eos>" and correct prediction
    # is exactly "" (empty string).  Both checks are strict — "<eos>" only on the
    # expected side, "" only on the predicted side.
    is_eos      = expected.strip() == "<eos>"
    pred_is_eos = predicted == ""

    # Turn type: first turn prompt contains "first prompt" or doesn't have conversation history
    is_first = ("first prompt" in inp.lower() or
                "generate the first" in inp.lower() or
                "FIRST_TURN" in inp)

    # Existing metrics
    token_ratio = raw_metrics.get("token_ratio", 0.0) or 0.0

    # For EOS examples the upstream BLEU/ROUGE were computed by comparing "" vs "<eos>"
    # and produce meaningless values.  Replace them with 1.0 (correct) or 0.0 (wrong).
    if is_eos:
        bleu_score        = 1.0 if pred_is_eos else 0.0
        rouge_l_f1        = 1.0 if pred_is_eos else 0.0
        rouge_l_precision = 1.0 if pred_is_eos else 0.0
        rouge_l_recall    = 1.0 if pred_is_eos else 0.0
        meteor            = 1.0 if pred_is_eos else 0.0
    else:
        bleu_score        = raw_metrics.get("bleu_score", 0.0) or 0.0
        rouge_l_f1        = raw_metrics.get("rouge_l_f1", 0.0) or 0.0
        rouge_l_precision = raw_metrics.get("rouge_l_precision", 0.0) or 0.0
        rouge_l_recall    = raw_metrics.get("rouge_l_recall", 0.0) or 0.0
        meteor            = compute_meteor(norm_exp, norm_pred)

    m = ExampleMetrics(
        example_idx             = example_idx,
        model_size              = model_size,
        model_variant           = variant,
        is_lora                 = is_lora,
        communication_style     = persona.get("communication_style", ""),
        emotional_state         = persona.get("emotional_state", ""),
        knowledge_level         = persona.get("knowledge_level", ""),
        goal_clarity            = persona.get("goal_clarity", ""),
        certainty               = scenario.get("certainty", ""),
        information_completeness= scenario.get("information_completeness", ""),
        expected_resolution     = scenario.get("expected_resolution", ""),
        merchant_name           = scenario.get("merchant_name", ""),
        amount                  = scenario.get("amount", ""),
        transaction_date        = scenario.get("transaction_date", ""),
        is_eos_example          = is_eos,
        is_first_turn           = is_first,
        loss                    = raw_metrics.get("loss", 0.0) or 0.0,
        perplexity              = raw_metrics.get("perplexity", 0.0) or 0.0,
        avg_token_confidence    = raw_metrics.get("avg_token_confidence", 0.0) or 0.0,
        bleu_score              = bleu_score,
        rouge_l_f1              = rouge_l_f1,
        rouge_l_precision       = rouge_l_precision,
        rouge_l_recall          = rouge_l_recall,
        exact_match             = bool(raw_metrics.get("exact_match", False)),
        token_ratio             = token_ratio,
        length_difference       = raw_metrics.get("length_difference", 0) or 0,
        predicted_tokens        = raw_metrics.get("predicted_tokens", 0) or 0,
        expected_tokens         = raw_metrics.get("expected_tokens", 0) or 0,
        generation_time_sec     = model_data.get("generation_time_seconds", 0.0) or 0.0,
        # New metrics (v1)
        normalized_exact_match  = pred_is_eos if is_eos else (norm_pred == norm_exp),
        eos_correct             = pred_is_eos if is_eos else None,
        eos_false_pos           = pred_is_eos if not is_eos else None,
        length_ratio_error      = abs(1.0 - token_ratio) if token_ratio > 0 else 1.0,
        style_compliance_score  = style_compliance(norm_pred, persona, conv_type),
        meteor_score            = meteor,
        # New metrics (v2 — entity)
        **entity_precision_recall(norm_exp, norm_pred),
        # New metrics (v2 — behavioural)
        ttr_predicted           = type_token_ratio(norm_pred),
        ttr_reference           = type_token_ratio(norm_exp),
        ttr_delta               = abs(type_token_ratio(norm_pred) - type_token_ratio(norm_exp)),
        hedge_rate_predicted    = compute_hedge_rate(norm_pred),
        hedge_rate_reference    = compute_hedge_rate(norm_exp),
        hedge_rate_delta        = abs(compute_hedge_rate(norm_pred) - compute_hedge_rate(norm_exp)),
        promise_rate_predicted  = compute_promise_rate(norm_pred),
        promise_rate_reference  = compute_promise_rate(norm_exp),
        promise_rate_delta      = abs(compute_promise_rate(norm_pred) - compute_promise_rate(norm_exp)),
        role_confused           = detect_role_confusion(norm_pred)["role_confused"],
        role_confusion_signals  = ", ".join(detect_role_confusion(norm_pred)["role_confusion_signals"]),
        # Conversation identity
        generation_idx          = meta.get("generation_idx"),
        # Probe / conversation-type metadata
        conversation_type       = conv_type,
        wrong_prior_belief      = wrong_prior,
        agent_failure_mode      = agent_fail_mode,
        user_caught_error       = caught_error,
        goal_completed          = goal_done,
        failure_mode            = failure_mode,
        # Turn-level probe signals (on predicted turn)
        is_pushback_turn        = (conv_type == "type_b" and check_pushback_turn(norm_pred)),
        prior_belief_expressed  = (conv_type == "type_a" and
                                   check_prior_belief_expressed(norm_pred, sc_idx)),
        # Raw text
        predicted_output        = predicted,
        expected_output         = expected,
    )
    return m


# ─────────────────────────────────────────────────────────────────────────────
#  Aggregation helpers
# ─────────────────────────────────────────────────────────────────────────────

def safe_mean(vals):
    v = [x for x in vals if x is not None]
    return round(np.mean(v), 4) if v else None


def aggregate(metrics_list: list) -> dict:
    """Return aggregate stats dict for a list of ExampleMetrics."""
    n = len(metrics_list)
    if n == 0:
        return {}

    eos_examples     = [m for m in metrics_list if m.is_eos_example]
    non_eos_examples = [m for m in metrics_list if not m.is_eos_example]

    eos_recall     = safe_mean([m.eos_correct   for m in eos_examples])
    eos_false_rate = safe_mean([m.eos_false_pos for m in non_eos_examples])

    return {
        "n":                    n,
        "n_eos":                len(eos_examples),
        "n_non_eos":            len(non_eos_examples),
        # Core metrics (non-EOS only for BLEU/ROUGE/METEOR — text comparison meaningless for EOS)
        "mean_bleu":            safe_mean([m.bleu_score   for m in non_eos_examples]),
        "mean_rouge_l_f1":      safe_mean([m.rouge_l_f1   for m in non_eos_examples]),
        "mean_meteor":          safe_mean([m.meteor_score  for m in non_eos_examples]),
        "mean_loss":            safe_mean([m.loss          for m in metrics_list]),
        "mean_perplexity":      safe_mean([m.perplexity    for m in metrics_list]),
        "norm_exact_match_rate":safe_mean([float(m.normalized_exact_match) for m in metrics_list]),
        # EOS behaviour
        "eos_recall":           eos_recall,      # None if no EOS examples in slice
        "eos_false_pos_rate":   eos_false_rate,
        # Length
        "mean_length_ratio_error": safe_mean([m.length_ratio_error for m in metrics_list]),
        "mean_token_ratio":        safe_mean([m.token_ratio         for m in metrics_list]),
        # Style
        "mean_style_compliance":   safe_mean([m.style_compliance_score for m in metrics_list]),
        # Speed
        "mean_gen_time_sec":       safe_mean([m.generation_time_sec for m in metrics_list]),
        # ── v2 metrics ──────────────────────────────────────────────────
        # Semantic (non-EOS only)
        "mean_bertscore_f1":       safe_mean([m.bertscore_f1        for m in non_eos_examples]),
        "mean_bertscore_precision":safe_mean([m.bertscore_precision  for m in non_eos_examples]),
        "mean_bertscore_recall":   safe_mean([m.bertscore_recall     for m in non_eos_examples]),
        "mean_bleurt":             safe_mean([m.bleurt_score         for m in non_eos_examples]),
        # Entity (non-EOS only)
        "mean_entity_precision":   safe_mean([m.entity_precision     for m in non_eos_examples]),
        "mean_entity_recall":      safe_mean([m.entity_recall        for m in non_eos_examples]),
        "mean_entity_f1":          safe_mean([m.entity_f1            for m in non_eos_examples]),
        # Behavioural
        "mean_ttr_delta":          safe_mean([m.ttr_delta            for m in non_eos_examples]),
        "mean_hedge_rate_delta":   safe_mean([m.hedge_rate_delta     for m in non_eos_examples]),
        "mean_promise_rate_pred":  safe_mean([m.promise_rate_predicted for m in non_eos_examples]),
        "mean_promise_rate_delta": safe_mean([m.promise_rate_delta   for m in non_eos_examples]),
        "role_confusion_rate":     safe_mean([float(m.role_confused) for m in metrics_list]),
        # LLM-as-judge
        "mean_judge_semantic":     safe_mean([m.judge_semantic_fidelity       for m in non_eos_examples]),
        "mean_judge_persona":      safe_mean([m.judge_persona_voice           for m in non_eos_examples]),
        "mean_judge_coherence":    safe_mean([m.judge_conversational_coherence for m in non_eos_examples]),
        "mean_judge_goal":         safe_mean([m.judge_goal_directedness       for m in non_eos_examples]),
        "mean_judge_realism":      safe_mean([m.judge_human_realism           for m in non_eos_examples]),
        "mean_judge_info_cal":     safe_mean([m.judge_information_calibration for m in non_eos_examples]),
        "mean_judge_overall":      safe_mean([m.judge_mean                    for m in non_eos_examples]),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Model comparison: win rates + Wilcoxon
# ─────────────────────────────────────────────────────────────────────────────

def compare_models(all_metrics: dict) -> list:
    """
    all_metrics: {variant_name: [ExampleMetrics, ...]} — must be same examples in same order.
    Returns list of dicts for comparison CSV.
    """
    variants = list(all_metrics.keys())
    rows = []

    for i, va in enumerate(variants):
        for vb in variants[i + 1:]:
            a_scores = [m.bleu_score for m in all_metrics[va]]
            b_scores = [m.bleu_score for m in all_metrics[vb]]

            n = len(a_scores)
            a_wins = sum(1 for a, b in zip(a_scores, b_scores) if a > b)
            b_wins = sum(1 for a, b in zip(a_scores, b_scores) if b > a)
            ties   = n - a_wins - b_wins

            # Wilcoxon signed-rank test (paired, non-parametric)
            diffs = [a - b for a, b in zip(a_scores, b_scores)]
            if any(d != 0 for d in diffs):
                try:
                    stat, p_val = wilcoxon(a_scores, b_scores)
                except Exception:
                    stat, p_val = float("nan"), float("nan")
            else:
                stat, p_val = 0.0, 1.0

            rows.append({
                "model_a":        va,
                "model_b":        vb,
                "n_examples":     n,
                "a_win_rate":     round(a_wins / n, 4),
                "b_win_rate":     round(b_wins / n, 4),
                "tie_rate":       round(ties   / n, 4),
                "a_wins":         a_wins,
                "b_wins":         b_wins,
                "ties":           ties,
                "mean_bleu_a":    round(np.mean(a_scores), 4),
                "mean_bleu_b":    round(np.mean(b_scores), 4),
                "wilcoxon_stat":  round(float(stat), 4) if not math.isnan(stat) else "nan",
                "p_value":        round(float(p_val), 6) if not math.isnan(p_val) else "nan",
                "significant":    (p_val < 0.05) if not math.isnan(p_val) else False,
            })

    return rows


# ─────────────────────────────────────────────────────────────────────────────
#  CSV writers
# ─────────────────────────────────────────────────────────────────────────────

def write_detailed_csv(all_flat: list, path: Path):
    if not all_flat:
        return
    fieldnames = [f for f in vars(ExampleMetrics()).keys()
                  if f not in ("predicted_output", "expected_output")]
    fieldnames += ["predicted_output", "expected_output"]  # keep at end
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in all_flat:
            row = asdict(m)
            writer.writerow({k: row[k] for k in fieldnames})
    print(f"  ✓  {path.name}  ({len(all_flat)} rows)")


def write_category_csv(all_metrics: dict, path: Path):
    rows = []
    variants = list(all_metrics.keys())
    for variant in variants:
        metrics = all_metrics[variant]
        # Overall
        agg = aggregate(metrics)
        agg.update({"model_variant": variant, "slice_dim": "overall", "slice_val": "all"})
        rows.append(agg)
        # Per slice
        for dim, dim_type in CATEGORY_SLICES:
            vals = sorted(set(getattr(m, dim) for m in metrics))
            for val in vals:
                subset = [m for m in metrics if getattr(m, dim) == val]
                agg2 = aggregate(subset)
                agg2.update({"model_variant": variant, "slice_dim": dim, "slice_val": val})
                rows.append(agg2)
        # EOS vs text split
        for label, filt in [("eos_examples", lambda m: m.is_eos_example),
                             ("text_examples", lambda m: not m.is_eos_example)]:
            subset = [m for m in metrics if filt(m)]
            agg3 = aggregate(subset)
            agg3.update({"model_variant": variant, "slice_dim": "turn_type", "slice_val": label})
            rows.append(agg3)

    if not rows:
        return
    fieldnames = ["model_variant", "slice_dim", "slice_val",
                  "n", "n_eos", "n_non_eos",
                  "mean_bleu", "mean_rouge_l_f1", "mean_meteor",
                  "mean_bertscore_f1", "mean_bertscore_precision", "mean_bertscore_recall",
                  "mean_bleurt",
                  "mean_loss", "mean_perplexity", "norm_exact_match_rate",
                  "eos_recall", "eos_false_pos_rate",
                  "mean_length_ratio_error", "mean_token_ratio",
                  "mean_style_compliance",
                  "mean_entity_precision", "mean_entity_recall", "mean_entity_f1",
                  "mean_ttr_delta", "mean_hedge_rate_delta",
                  "mean_promise_rate_pred", "mean_promise_rate_delta",
                  "role_confusion_rate",
                  "mean_judge_semantic", "mean_judge_persona", "mean_judge_coherence",
                  "mean_judge_goal", "mean_judge_realism", "mean_judge_info_cal",
                  "mean_judge_overall",
                  "mean_gen_time_sec"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✓  {path.name}  ({len(rows)} rows)")


def write_comparison_csv(rows: list, path: Path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✓  {path.name}  ({len(rows)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
#  Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, name: str):
    IMAGES_DIR.mkdir(exist_ok=True)
    path = IMAGES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {path.name}")


# ── Figure 13 — Category Heatmap ─────────────────────────────────────────────
def fig_category_heatmap(all_metrics: dict):
    """
    Grid of heatmaps: one per persona attribute.
    Rows = attribute values, columns = model variants.
    Cell = mean BLEU on non-EOS examples.
    Answers: "is terse behaviour harder for smaller models?"
    """
    _set_style()
    # Only show LoRA variants to reduce clutter; base shown in efficiency plot
    lora_variants = [v for v in all_metrics if v.startswith("lora_")]
    if not lora_variants:
        lora_variants = list(all_metrics.keys())

    slice_dims = [
        ("communication_style", "Communication Style"),
        ("emotional_state",     "Emotional State"),
        ("knowledge_level",     "Knowledge Level"),
        ("goal_clarity",        "Goal Clarity"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        "Mean BLEU Score by Persona Category  ×  Model Variant\n"
        "Which persona types does each model size handle best?  (non-EOS examples only)",
        fontsize=14, fontweight="bold", y=1.02,
    )

    for ax, (dim, dim_label) in zip(axes.flat, slice_dims):
        attr_vals = sorted(set(getattr(m, dim) for mlist in all_metrics.values()
                               for m in mlist))
        matrix = []
        for av in attr_vals:
            row = []
            for variant in lora_variants:
                subset = [m for m in all_metrics[variant]
                          if getattr(m, dim) == av and not m.is_eos_example]
                row.append(round(np.mean([m.bleu_score for m in subset]), 3)
                            if subset else 0.0)
            matrix.append(row)

        mat = np.array(matrix)
        im  = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03,
                     label="Mean BLEU (0–1)")

        col_labels = [v.replace("_", " ").upper() for v in lora_variants]
        ax.set_xticks(range(len(lora_variants)))
        ax.set_xticklabels(col_labels, fontsize=10, fontweight="bold")
        ax.set_yticks(range(len(attr_vals)))
        row_labels = []
        for av in attr_vals:
            n = sum(1 for m in all_metrics[lora_variants[0]]
                    if getattr(m, dim) == av and not m.is_eos_example)
            row_labels.append(f"{av.replace('_', ' ')}  (n={n})")
        ax.set_yticklabels(row_labels, fontsize=10)
        ax.set_title(f"By {dim_label}", pad=8)

        for i in range(len(attr_vals)):
            for j in range(len(lora_variants)):
                val = mat[i, j]
                text_color = "black" if 0.25 < val < 0.78 else "white"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=text_color)

    fig.tight_layout()
    _save(fig, "13_category_heatmap.png")


# ── Figure 14 — Model Comparison ─────────────────────────────────────────────
def fig_model_comparison(all_metrics: dict, comparison_rows: list):
    """
    Left: pairwise win-rate matrix (LoRA models only).
    Right: bar chart of Wilcoxon p-values for each LoRA pair.
    """
    _set_style()

    lora_variants = sorted(v for v in all_metrics if v.startswith("lora_"))
    if len(lora_variants) < 2:
        print("  ⚠  Need ≥2 LoRA variants for model comparison figure — skipping.")
        return

    # Build N×N win-rate matrix for LoRA models
    n = len(lora_variants)
    win_matrix = np.full((n, n), np.nan)
    for row in comparison_rows:
        a, b = row["model_a"], row["model_b"]
        if a in lora_variants and b in lora_variants:
            ia, ib = lora_variants.index(a), lora_variants.index(b)
            win_matrix[ia, ib] = row["a_win_rate"]
            win_matrix[ib, ia] = row["b_win_rate"]
    # Diagonal = mean BLEU
    for i, v in enumerate(lora_variants):
        vals = [m.bleu_score for m in all_metrics[v] if not m.is_eos_example]
        win_matrix[i, i] = np.mean(vals) if vals else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "Cross-Model Comparison  —  LoRA Variants\n"
        "Win rate = fraction of 139 examples where model A has higher BLEU than model B",
        fontsize=14, fontweight="bold", y=1.02,
    )

    # ── Left: win-rate matrix ─────────────────────────────────────────
    ax1 = axes[0]
    labels = [v.replace("_", " ").upper() for v in lora_variants]
    # Custom colormap: diagonal in blue (mean BLEU), off-diagonal in RdYlGn (win rate)
    im = ax1.imshow(win_matrix, vmin=0, vmax=1, cmap="RdYlGn")
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04,
                 label="Win rate  (diagonal = mean BLEU)")
    ax1.set_xticks(range(n)); ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_yticks(range(n)); ax1.set_yticklabels(labels, fontsize=11)
    ax1.set_title("Pairwise Win Rate Matrix\n(row model vs column model)", pad=10)

    for i in range(n):
        for j in range(n):
            val = win_matrix[i, j]
            if np.isnan(val):
                continue
            if i == j:
                label_str = f"μBLEU\n{val:.3f}"
            else:
                label_str = f"{val:.1%}"
            text_color = "black" if 0.25 < val < 0.75 else "white"
            ax1.text(j, i, label_str, ha="center", va="center",
                     fontsize=9.5, fontweight="bold", color=text_color)

    # Highlight diagonal differently
    for i in range(n):
        ax1.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                    fill=False, edgecolor="#1B3A4B",
                                    linewidth=2.5, zorder=5))

    # ── Right: Wilcoxon p-values ──────────────────────────────────────
    ax2 = axes[1]
    lora_rows = [r for r in comparison_rows
                 if r["model_a"] in lora_variants and r["model_b"] in lora_variants]

    pair_labels = [f"{r['model_a'].replace('_',' ').upper()}\nvs\n{r['model_b'].replace('_',' ').upper()}"
                   for r in lora_rows]
    mean_diff   = [round(r["mean_bleu_a"] - r["mean_bleu_b"], 4) for r in lora_rows]
    p_vals      = [r["p_value"] if isinstance(r["p_value"], float) else float("nan")
                   for r in lora_rows]

    bar_colors = ["#2A9D8F" if d > 0 else "#E76F51" for d in mean_diff]
    bars = ax2.barh(range(len(lora_rows)), mean_diff, color=bar_colors,
                    edgecolor="white", linewidth=0.7, height=0.5)

    for i, (bar, pv) in enumerate(zip(bars, p_vals)):
        sig_str = "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else "ns"))
        xpos = bar.get_width() + 0.002 if bar.get_width() >= 0 else bar.get_width() - 0.002
        ha   = "left" if bar.get_width() >= 0 else "right"
        ax2.text(xpos, i, f"p={pv:.4f} {sig_str}", va="center", ha=ha,
                 fontsize=9, color="#333333")

    ax2.axvline(0, color="#333333", linewidth=1.2, linestyle="-")
    ax2.set_yticks(range(len(lora_rows)))
    ax2.set_yticklabels(pair_labels, fontsize=9)
    ax2.set_xlabel("BLEU difference (row A − row B)")
    ax2.set_title("Mean BLEU Difference + Wilcoxon Significance\n"
                  "(* p<0.05  ** p<0.01  *** p<0.001  ns = not significant)", pad=10)

    fig.tight_layout()
    _save(fig, "14_model_comparison.png")


# ── Figure 15 — EOS Analysis ─────────────────────────────────────────────────
def fig_eos_analysis(all_metrics: dict):
    """
    Three panels:
    1. EOS recall per model (of 16 expected-EOS examples, how many predicted ""?)
    2. EOS false positive rate per model (of 123 text examples, how many predicted ""?)
    3. For EOS examples: what did each model actually predict?  (qualitative bar)
    """
    _set_style()
    all_variants = list(all_metrics.keys())

    # Panel 1 data
    eos_recall = {}
    eos_fp     = {}
    for variant, metrics in all_metrics.items():
        eos_ex  = [m for m in metrics if m.is_eos_example]
        text_ex = [m for m in metrics if not m.is_eos_example]
        eos_recall[variant] = (
            sum(m.eos_correct for m in eos_ex) / len(eos_ex) if eos_ex else 0.0
        )
        eos_fp[variant] = (
            sum(m.eos_false_pos for m in text_ex) / len(text_ex) if text_ex else 0.0
        )

    # Panel 3: length distribution of predictions on EOS examples
    eos_pred_lengths = {
        variant: [word_count(normalize(m.predicted_output))
                  for m in metrics if m.is_eos_example]
        for variant, metrics in all_metrics.items()
    }

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle(
        "EOS Prediction Analysis — Does the model learn when to end the conversation?\n"
        "Expected output is <eos> (empty) for 16/139 examples (11.5% of val set)",
        fontsize=13, fontweight="bold", y=1.03,
    )

    variant_display = [v.replace("_", " ").upper() for v in all_variants]

    # ── 15a: EOS recall ───────────────────────────────────────────────
    ax1 = axes[0]
    recalls = [eos_recall[v] for v in all_variants]
    colors  = [MODEL_COLORS.get(v, "#888") for v in all_variants]
    bars = ax1.bar(range(len(all_variants)), recalls, color=colors,
                   edgecolor="white", linewidth=0.7, width=0.6)
    for bar, val in zip(bars, recalls):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.0%}", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")
    ax1.set_xticks(range(len(all_variants)))
    ax1.set_xticklabels(variant_display, fontsize=9, rotation=20, ha="right")
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("EOS recall (fraction of 16 EOS examples predicted correctly)")
    ax1.set_title("EOS Recall per Model\n"
                  "Did model predict '' when expected output was <eos>?")
    ax1.axhline(1.0, color="#333", linewidth=0.8, linestyle=":", alpha=0.5)

    # ── 15b: EOS false positive rate ─────────────────────────────────
    ax2 = axes[1]
    fp_rates = [eos_fp[v] for v in all_variants]
    bars2 = ax2.bar(range(len(all_variants)), fp_rates, color=colors,
                    edgecolor="white", linewidth=0.7, width=0.6)
    for bar, val in zip(bars2, fp_rates):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.1%}", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")
    ax2.set_xticks(range(len(all_variants)))
    ax2.set_xticklabels(variant_display, fontsize=9, rotation=20, ha="right")
    ax2.set_ylim(0, max(fp_rates) * 1.3 + 0.05)
    ax2.set_ylabel("False positive rate (fraction of 123 text examples predicted as '')")
    ax2.set_title("EOS False Positive Rate\n"
                  "Did model wrongly predict '' when text was expected?")

    # ── 15c: word-count distribution of EOS-example predictions ──────
    ax3 = axes[2]
    positions = range(len(all_variants))
    bp = ax3.boxplot(
        [eos_pred_lengths[v] for v in all_variants],
        positions=list(positions),
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="white", linewidth=2),
    )
    for patch, v in zip(bp["boxes"], all_variants):
        patch.set_facecolor(MODEL_COLORS.get(v, "#888"))

    ax3.axhline(0, color="#C1121F", linewidth=1.6, linestyle="--",
                label="Target: 0 words (predict EOS)")
    ax3.set_xticks(list(positions))
    ax3.set_xticklabels(variant_display, fontsize=9, rotation=20, ha="right")
    ax3.set_ylabel("Word count of prediction on EOS-expected examples")
    ax3.set_title("Prediction Length on EOS Examples\n"
                  "0 = correct (model predicted nothing)")
    ax3.legend(fontsize=9)

    fig.tight_layout()
    _save(fig, "15_eos_analysis.png")


# ── Figure 16 — Efficiency Frontier ──────────────────────────────────────────
def fig_efficiency_frontier(all_metrics: dict):
    """
    Scatter: mean BLEU vs model size (params).
    Bubble area ∝ mean generation time.
    Separate lines for base and LoRA.
    """
    _set_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Efficiency Frontier — Quality vs Model Scale\n"
        "Does bigger always mean better?  Bubble size = mean generation time",
        fontsize=14, fontweight="bold", y=1.02,
    )

    ax1, ax2 = axes

    for variant, metrics in all_metrics.items():
        size_key = variant.split("_", 1)[1] if "_" in variant else variant
        model_name = f"gemma_{size_key}"
        params = MODEL_INFO.get(model_name, {}).get("params_b", None)
        if params is None:
            continue

        non_eos = [m for m in metrics if not m.is_eos_example]
        mean_bleu = np.mean([m.bleu_score for m in non_eos]) if non_eos else 0.0
        mean_rouge = np.mean([m.rouge_l_f1 for m in non_eos]) if non_eos else 0.0
        mean_time = np.mean([m.generation_time_sec for m in metrics]) if metrics else 0.0

        is_lora = variant.startswith("lora_")
        color   = MODEL_COLORS.get(variant, "#888")
        marker  = "^" if is_lora else "o"
        label   = variant.replace("_", " ").upper()

        # BLEU plot
        ax1.scatter(params, mean_bleu, s=mean_time * 120, color=color,
                    marker=marker, alpha=0.85, edgecolors="white",
                    linewidths=1.5, zorder=3, label=label)
        ax1.annotate(label, (params, mean_bleu),
                     textcoords="offset points", xytext=(6, 4),
                     fontsize=8.5, color=color)

        # ROUGE plot
        ax2.scatter(params, mean_rouge, s=mean_time * 120, color=color,
                    marker=marker, alpha=0.85, edgecolors="white",
                    linewidths=1.5, zorder=3)
        ax2.annotate(label, (params, mean_rouge),
                     textcoords="offset points", xytext=(6, 4),
                     fontsize=8.5, color=color)

    # Draw trend lines for LoRA and base separately
    for ax, metric_name in [(ax1, "BLEU"), (ax2, "ROUGE-L F1")]:
        for variant_type, style in [("lora", "--"), ("base", ":")]:
            type_variants = [v for v in all_metrics if v.startswith(variant_type)]
            xs, ys = [], []
            for v in type_variants:
                size_key = v.split("_", 1)[1]
                params = MODEL_INFO.get(f"gemma_{size_key}", {}).get("params_b")
                if params is None:
                    continue
                non_eos = [m for m in all_metrics[v] if not m.is_eos_example]
                if not non_eos:
                    continue
                y = np.mean([m.bleu_score if metric_name == "BLEU" else m.rouge_l_f1
                              for m in non_eos])
                xs.append(params); ys.append(y)
            if len(xs) >= 2:
                order = sorted(range(len(xs)), key=lambda i: xs[i])
                ax.plot([xs[i] for i in order], [ys[i] for i in order],
                        color="#555", linewidth=1.2, linestyle=style, alpha=0.6,
                        label=f"{variant_type} trend")

        ax.set_xlabel("Model size (billion parameters)")
        ax.set_ylabel(f"Mean {metric_name} (non-EOS examples)")
        ax.set_title(f"{metric_name} vs Model Scale\n(▲ = LoRA  ●= base  bubble = gen time)")
        ax.set_xticks([4, 12, 27])
        ax.legend(fontsize=8, loc="lower right")

    # Shared bubble-size legend
    for ax in axes:
        for t, sz in [(f"1s gen time", 120), (f"5s gen time", 600)]:
            ax.scatter([], [], s=sz, c="#aaa", alpha=0.5, label=t)
        ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    _save(fig, "16_efficiency_frontier.png")


# ── Figure 17 — Style Compliance ─────────────────────────────────────────────
def fig_style_compliance(all_metrics: dict):
    """
    Two panels:
    1. Grouped bars: style compliance score by communication_style × model.
    2. Box plots: token ratio error by communication_style (are terse predictions the right length?).
    """
    _set_style()
    lora_variants = [v for v in all_metrics if v.startswith("lora_")]
    if not lora_variants:
        lora_variants = list(all_metrics.keys())

    styles  = ["terse", "direct", "indirect"]
    n_styles = len(styles)
    x       = np.arange(n_styles)
    w       = 0.8 / len(lora_variants)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "Style Compliance — Does the model honour the communication style persona?\n"
        "Terse = short responses  ·  Direct = no hedging  ·  Indirect = hedged, contextual",
        fontsize=14, fontweight="bold", y=1.02,
    )

    # ── 17a: grouped compliance bars ─────────────────────────────────
    ax1 = axes[0]
    for i, variant in enumerate(lora_variants):
        means = []
        cis   = []
        for style in styles:
            subset = [m for m in all_metrics[variant]
                      if m.communication_style == style and not m.is_eos_example]
            if subset:
                vals = [m.style_compliance_score for m in subset]
                means.append(np.mean(vals))
                cis.append(np.std(vals) / math.sqrt(len(vals)))
            else:
                means.append(0.0)
                cis.append(0.0)

        offset = (i - len(lora_variants) / 2 + 0.5) * w
        bars = ax1.bar(x + offset, means, w * 0.9,
                       label=variant.replace("_", " ").upper(),
                       color=MODEL_COLORS.get(variant, "#888"),
                       edgecolor="white", linewidth=0.6)
        ax1.errorbar(x + offset, means, yerr=cis, fmt="none",
                     ecolor="#333", elinewidth=1, capsize=3)

        for bar, val in zip(bars, means):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    # Count labels
    for j, style in enumerate(styles):
        n = sum(1 for m in all_metrics[lora_variants[0]]
                if m.communication_style == style and not m.is_eos_example)
        ax1.text(j, -0.08, f"n={n}", ha="center", fontsize=8.5,
                 color="#666", transform=ax1.get_xaxis_transform())

    ax1.set_xticks(x)
    ax1.set_xticklabels([s.capitalize() for s in styles])
    ax1.set_ylabel("Mean style compliance score (0–1)")
    ax1.set_ylim(0, 1.2)
    ax1.set_title("Style Compliance by Communication Style\n(error bars = standard error)")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.axhline(1.0, color="#333", linewidth=0.8, linestyle=":", alpha=0.5)

    # ── 17b: token ratio error by style ──────────────────────────────
    ax2 = axes[1]
    # Show distribution of length ratio error per style for LoRA vs base
    positions_base = []
    all_data   = []
    all_colors = []
    all_labels = []
    tick_positions = []
    tick_labels    = []
    pos = 0

    for style in styles:
        tick_mid = []
        for variant in sorted(all_metrics.keys()):   # all variants here for comparison
            subset = [m for m in all_metrics[variant]
                      if m.communication_style == style and not m.is_eos_example]
            if not subset:
                continue
            vals = [m.length_ratio_error for m in subset]
            all_data.append(vals)
            all_colors.append(MODEL_COLORS.get(variant, "#888"))
            all_labels.append(variant.replace("_", " ").upper())
            tick_mid.append(pos)
            pos += 1
        if tick_mid:
            tick_positions.append(np.mean(tick_mid))
            tick_labels.append(style.capitalize())
        pos += 0.5   # gap between style groups

    bp = ax2.boxplot(all_data, positions=list(range(len(all_data))),
                     patch_artist=True, widths=0.55,
                     medianprops=dict(color="white", linewidth=2))
    for patch, col in zip(bp["boxes"], all_colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)

    ax2.axhline(0, color="#C1121F", linewidth=1.6, linestyle="--",
                label="0 = perfect length match")
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, fontsize=11)
    ax2.set_ylabel("Length ratio error  |1 − predicted_tokens/expected_tokens|")
    ax2.set_title("Length Ratio Error by Communication Style\n"
                  "(0 = perfect length match with expected output)")
    ax2.legend(fontsize=9)

    # Add variant legend patches
    legend_items = [mpatches.Patch(color=MODEL_COLORS.get(v, "#888"),
                                   label=v.replace("_", " ").upper())
                    for v in sorted(all_metrics.keys())]
    ax2.legend(handles=legend_items, fontsize=8, loc="upper right")

    fig.tight_layout()
    _save(fig, "17_style_compliance.png")


# ─────────────────────────────────────────────────────────────────────────────
#  Terminal summary
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(val, width=6, fmt=".3f"):
    """Format a value that may be None."""
    if val is None:
        return f"{'n/a':>{width}}"
    return f"{val:>{width}{fmt}}"


def print_summary(all_metrics: dict):
    sep = "─" * 100
    print(f"\n{'=' * 100}")
    print("  PREDICTION EVALUATION SUMMARY")
    print(f"{'=' * 100}")

    # ── Table 1: Core surface metrics ─────────────────────────────────────
    print(f"\n{sep}")
    print("  TABLE 1: Surface Metrics  (non-EOS examples only for text metrics)")
    print(f"  {'Model':<18} {'N':>4}  {'BLEU':>6}  {'ROUGE':>6}  {'METEOR':>6}  "
          f"{'BERT-F1':>7}  {'BLEURT':>7}  {'NormEM%':>7}  {'EOS-Rec':>8}  {'StyleC':>6}")
    print(f"  {'─'*18}  {'─'*4}  {'─'*6}  {'─'*6}  {'─'*6}  "
          f"{'─'*7}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*6}")

    for variant in sorted(all_metrics.keys()):
        agg = aggregate(all_metrics[variant])
        print(f"  {variant:<18} {agg['n']:>4}  "
              f"{_fmt(agg['mean_bleu'])}  "
              f"{_fmt(agg['mean_rouge_l_f1'])}  "
              f"{_fmt(agg['mean_meteor'])}  "
              f"{_fmt(agg['mean_bertscore_f1'], 7)}  "
              f"{_fmt(agg['mean_bleurt'], 7)}  "
              f"{(agg['norm_exact_match_rate'] or 0)*100:>6.1f}%  "
              f"{(agg['eos_recall'] or 0):>7.1%}  "
              f"{_fmt(agg['mean_style_compliance'])}")

    # ── Table 2: Behavioural signals ──────────────────────────────────────
    print(f"\n{sep}")
    print("  TABLE 2: Behavioural Signals  (lower delta = better calibration)")
    print(f"  {'Model':<18} {'EntF1':>6}  {'TTR-Δ':>6}  {'Hedge-Δ':>8}  "
          f"{'PromΔ':>6}  {'RoleCon%':>8}")
    print(f"  {'─'*18}  {'─'*6}  {'─'*6}  {'─'*8}  {'─'*6}  {'─'*8}")

    for variant in sorted(all_metrics.keys()):
        agg = aggregate(all_metrics[variant])
        rc = agg.get('role_confusion_rate')
        print(f"  {variant:<18} "
              f"{_fmt(agg['mean_entity_f1'])}  "
              f"{_fmt(agg['mean_ttr_delta'])}  "
              f"{_fmt(agg['mean_hedge_rate_delta'], 8)}  "
              f"{_fmt(agg['mean_promise_rate_delta'])}  "
              f"{(rc or 0)*100:>7.1f}%")

    # ── Table 3: LLM-as-judge (if available) ──────────────────────────────
    sample_agg = aggregate(list(all_metrics.values())[0])
    if sample_agg.get("mean_judge_overall") is not None:
        print(f"\n{sep}")
        print("  TABLE 3: LLM-as-Judge Scores  (1–5, higher = better)")
        print(f"  {'Model':<18} {'Seman':>6}  {'Perso':>6}  {'Coher':>6}  "
              f"{'Goal':>6}  {'Real':>6}  {'InfoC':>6}  {'Mean':>6}")
        print(f"  {'─'*18}  {'─'*6}  {'─'*6}  {'─'*6}  "
              f"{'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}")
        for variant in sorted(all_metrics.keys()):
            agg = aggregate(all_metrics[variant])
            print(f"  {variant:<18} "
                  f"{_fmt(agg['mean_judge_semantic'])}  "
                  f"{_fmt(agg['mean_judge_persona'])}  "
                  f"{_fmt(agg['mean_judge_coherence'])}  "
                  f"{_fmt(agg['mean_judge_goal'])}  "
                  f"{_fmt(agg['mean_judge_realism'])}  "
                  f"{_fmt(agg['mean_judge_info_cal'])}  "
                  f"{_fmt(agg['mean_judge_overall'])}")

    # ── Table 4: Probe type breakdown (if probe data present) ─────────────
    has_probes = any(
        m.conversation_type != "success"
        for mlist in all_metrics.values()
        for m in mlist
    )
    if has_probes:
        print(f"\n{sep}")
        print("  TABLE 4: Metrics by Conversation Type  (LoRA variants, non-EOS examples)")
        print(f"  {'Model':<18} {'Type':<10} {'N':>4}  {'BLEU':>6}  {'StyleC':>6}  "
              f"{'PushbkRate':>11}  {'PriorRate':>10}")
        print(f"  {'─'*18}  {'─'*10}  {'─'*4}  {'─'*6}  {'─'*6}  {'─'*11}  {'─'*10}")
        for variant in sorted(m for m in all_metrics if m.startswith("lora_")):
            for ctype in ["success", "type_a", "type_b"]:
                subset = [m for m in all_metrics[variant]
                          if m.conversation_type == ctype and not m.is_eos_example]
                if not subset:
                    continue
                bleu = safe_mean([m.bleu_score for m in subset]) or 0
                stylec = safe_mean([m.style_compliance_score for m in subset]) or 0
                pushbk = safe_mean([float(m.is_pushback_turn) for m in subset]) or 0
                prior  = safe_mean([float(m.prior_belief_expressed) for m in subset]) or 0
                print(f"  {variant:<18} {ctype:<10} {len(subset):>4}  "
                      f"{bleu:>6.3f}  {stylec:>6.3f}  "
                      f"{pushbk:>10.1%}  {prior:>9.1%}")

    # ── EOS analysis ──────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  EOS ANALYSIS")
    print(sep)
    for variant in sorted(all_metrics.keys()):
        metrics = all_metrics[variant]
        eos_ex  = [m for m in metrics if m.is_eos_example]
        correct = sum(m.eos_correct for m in eos_ex)
        wrong   = [m for m in eos_ex if not m.eos_correct]
        print(f"  {variant:<20}: {correct}/{len(eos_ex)} correct EOS predictions")
        for m in wrong[:3]:
            print(f"    ✗ Conv {m.example_idx}: predicted → \"{m.predicted_output[:60]}\"")

    # ── Role confusion flagged examples ───────────────────────────────────
    print(f"\n{sep}")
    print("  ROLE CONFUSION  (predicted user turns containing assistant-side language)")
    print(sep)
    for variant in sorted(all_metrics.keys()):
        confused = [m for m in all_metrics[variant] if m.role_confused]
        print(f"  {variant:<20}: {len(confused)} turns flagged")
        for m in confused[:3]:
            print(f"    ⚠ #{m.example_idx}: [{m.role_confusion_signals}] "
                  f"→ \"{m.predicted_output[:60]}\"")

    print(f"{'=' * 100}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Conversation-level metrics  (grouped by generation_idx)
# ─────────────────────────────────────────────────────────────────────────────

def compute_conversation_metrics(all_metrics: dict) -> list:
    """
    Group turns by (variant, generation_idx), compute conversation-level metrics.
    Returns list of dicts for conversation_metrics.csv.
    """
    rows = []
    for variant, metrics in all_metrics.items():
        # Group by generation_idx
        conv_groups = defaultdict(list)
        for m in metrics:
            if m.generation_idx is not None:
                conv_groups[m.generation_idx].append(m)

        for gen_idx, turns in sorted(conv_groups.items()):
            turns_sorted = sorted(turns, key=lambda t: t.example_idx)
            non_eos = [t for t in turns_sorted if not t.is_eos_example]

            # Style compliance across turns
            style_scores = [t.style_compliance_score for t in non_eos]
            style_mean = np.mean(style_scores) if style_scores else None
            style_std  = np.std(style_scores)  if len(style_scores) > 1 else 0.0
            style_min  = min(style_scores)      if style_scores else None

            # BERTScore conversation average
            bert_scores = [t.bertscore_f1 for t in non_eos if t.bertscore_f1 is not None]
            bert_mean = round(np.mean(bert_scores), 4) if bert_scores else None

            # Hedge rate trajectory: Spearman correlation between turn position and hedge rate
            if len(non_eos) >= 3:
                pred_hedges = [t.hedge_rate_predicted for t in non_eos]
                ref_hedges  = [t.hedge_rate_reference for t in non_eos]
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        hedge_corr, _ = spearmanr(pred_hedges, ref_hedges)
                    hedge_corr = round(hedge_corr, 4) if not np.isnan(hedge_corr) else None
                except Exception:
                    hedge_corr = None
            else:
                hedge_corr = None

            # Length trajectory fidelity: Spearman between predicted & reference lengths
            if len(non_eos) >= 3:
                pred_lens = [t.predicted_tokens for t in non_eos]
                ref_lens  = [t.expected_tokens  for t in non_eos]
                try:
                    len_corr, _ = spearmanr(pred_lens, ref_lens)
                    len_corr = round(len_corr, 4) if not np.isnan(len_corr) else None
                except Exception:
                    len_corr = None
            else:
                len_corr = None

            # Conversation-level hedge rate delta
            conv_hedge_pred = np.mean([t.hedge_rate_predicted for t in non_eos]) if non_eos else 0
            conv_hedge_ref  = np.mean([t.hedge_rate_reference for t in non_eos]) if non_eos else 0

            # Conversation-level promise rate
            conv_promise_pred = np.mean([t.promise_rate_predicted for t in non_eos]) if non_eos else 0

            # Any role confusion in this conversation?
            n_role_confused = sum(1 for t in turns_sorted if t.role_confused)

            # LLM judge conversation average
            judge_scores = [t.judge_mean for t in non_eos if t.judge_mean is not None]
            judge_conv_mean = round(np.mean(judge_scores), 4) if judge_scores else None

            # Self-BLEU across predicted turns in this conversation (mode collapse signal)
            pred_texts = [t.predicted_output for t in non_eos if t.predicted_output.strip()]
            self_bleu = compute_self_bleu(pred_texts) if len(pred_texts) >= 2 else None

            # Persona info + conversation type
            persona_info = {}
            conv_type_val = "success"
            if turns_sorted:
                t0 = turns_sorted[0]
                persona_info = {
                    "communication_style": t0.communication_style,
                    "emotional_state": t0.emotional_state,
                    "knowledge_level": t0.knowledge_level,
                    "goal_clarity": t0.goal_clarity,
                }
                conv_type_val = t0.conversation_type

            # type_b: did any predicted turn catch the error? (compare ref signal)
            pushback_pred = sum(1 for t in non_eos if t.is_pushback_turn)
            # type_b: reference pushback rate (recomputed on expected_output side)
            pushback_ref = sum(
                1 for t in non_eos
                if check_pushback_turn(normalize(t.expected_output))
            ) if conv_type_val == "type_b" else 0
            # type_a: prior belief present in predicted turns
            prior_belief_pred = sum(1 for t in non_eos if t.prior_belief_expressed)

            rows.append({
                "model_variant": variant,
                "generation_idx": gen_idx,
                "conversation_type": conv_type_val,
                "n_turns": len(turns_sorted),
                "n_text_turns": len(non_eos),
                **persona_info,
                # Style coherence
                "style_compliance_mean": round(style_mean, 4) if style_mean is not None else None,
                "style_compliance_std": round(style_std, 4),
                "style_compliance_min": round(style_min, 4) if style_min is not None else None,
                # Semantic
                "bertscore_f1_mean": bert_mean,
                # Trajectory
                "hedge_rate_correlation": hedge_corr,
                "length_trajectory_correlation": len_corr,
                "hedge_rate_delta": round(abs(conv_hedge_pred - conv_hedge_ref), 4),
                "promise_rate_predicted": round(conv_promise_pred, 4),
                # Behavioural
                "n_role_confused_turns": n_role_confused,
                "self_bleu": self_bleu,
                # Judge
                "judge_mean": judge_conv_mean,
                # Probe signals (type_a / type_b — 0 for success)
                "pushback_turns_pred": pushback_pred,
                "pushback_turns_ref": pushback_ref,
                "pushback_missed": (conv_type_val == "type_b" and pushback_ref > 0
                                    and pushback_pred == 0),
                "prior_belief_turns_pred": prior_belief_pred,
            })

    return rows


def write_conversation_csv(rows: list, path: Path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✓  {path.name}  ({len(rows)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
#  New visualisations  (v2 metrics)
# ─────────────────────────────────────────────────────────────────────────────

def fig_bertscore_comparison(all_metrics: dict):
    """Grouped bar chart: mean BERTScore F1 vs BLEU vs ROUGE per model variant."""
    _set_style()
    lora_variants = sorted(v for v in all_metrics if v.startswith("lora_"))
    if not lora_variants:
        lora_variants = sorted(all_metrics.keys())

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(
        "Semantic Metrics Comparison — BERTScore vs BLEU vs ROUGE\n"
        "BERTScore captures paraphrase similarity that n-gram metrics miss",
        fontsize=13, fontweight="bold", y=1.02,
    )

    metric_names = ["BLEU", "ROUGE-L", "METEOR", "BERTScore F1"]
    x = np.arange(len(lora_variants))
    w = 0.18

    for i, (mname, mkey) in enumerate([
        ("BLEU", "mean_bleu"), ("ROUGE-L", "mean_rouge_l_f1"),
        ("METEOR", "mean_meteor"), ("BERTScore F1", "mean_bertscore_f1"),
    ]):
        vals = []
        for v in lora_variants:
            agg = aggregate(all_metrics[v])
            vals.append(agg.get(mkey) or 0.0)
        offset = (i - 1.5) * w
        bars = ax.bar(x + offset, vals, w * 0.9, label=mname, alpha=0.85)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([v.replace("_", " ").upper() for v in lora_variants])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=10)
    ax.set_title("All variants — non-EOS examples only")
    fig.tight_layout()
    _save(fig, "18_bertscore_comparison.png")


def fig_behavioural_signals(all_metrics: dict):
    """Behavioural signal comparison: hedge delta, promise rate, TTR delta, role confusion."""
    _set_style()
    all_variants = sorted(all_metrics.keys())
    lora_variants = [v for v in all_variants if v.startswith("lora_")]
    base_variants = [v for v in all_variants if v.startswith("base_")]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        "Behavioural Signal Analysis\n"
        "Detecting LLM-like failure modes: hedge gap, promise language, lexical collapse, role confusion",
        fontsize=14, fontweight="bold", y=1.02,
    )

    # 19a: Hedge rate delta (base vs LoRA)
    ax = axes[0, 0]
    variants_to_plot = lora_variants if lora_variants else all_variants
    hedge_deltas = []
    for v in variants_to_plot:
        non_eos = [m for m in all_metrics[v] if not m.is_eos_example]
        hedge_deltas.append([m.hedge_rate_delta for m in non_eos])
    bp = ax.boxplot(hedge_deltas, patch_artist=True, widths=0.5,
                    medianprops=dict(color="white", linewidth=2))
    for patch, v in zip(bp["boxes"], variants_to_plot):
        patch.set_facecolor(MODEL_COLORS.get(v, "#888"))
    ax.set_xticklabels([v.replace("_"," ").upper() for v in variants_to_plot], fontsize=9)
    ax.set_ylabel("|hedge_rate(pred) − hedge_rate(ref)|")
    ax.set_title("Hedge Rate Calibration\n(lower = better match to reference hedging)")
    ax.axhline(0, color="#C1121F", linewidth=1, linestyle="--", alpha=0.5)

    # 19b: Promise language rate (pred only — should be near-zero for good user sim)
    ax = axes[0, 1]
    promise_rates = []
    for v in all_variants:
        non_eos = [m for m in all_metrics[v] if not m.is_eos_example]
        promise_rates.append([m.promise_rate_predicted for m in non_eos])
    bp = ax.boxplot(promise_rates, patch_artist=True, widths=0.5,
                    medianprops=dict(color="white", linewidth=2))
    for patch, v in zip(bp["boxes"], all_variants):
        patch.set_facecolor(MODEL_COLORS.get(v, "#888"))
    ax.set_xticklabels([v.replace("_"," ").upper() for v in all_variants],
                       fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("Promise phrase rate")
    ax.set_title("Promise Language  (Wang et al. 2025)\n"
                 "(high = LLM-like over-commitment)")

    # 19c: TTR delta
    ax = axes[1, 0]
    ttr_deltas = []
    for v in variants_to_plot:
        non_eos = [m for m in all_metrics[v] if not m.is_eos_example]
        ttr_deltas.append([m.ttr_delta for m in non_eos])
    bp = ax.boxplot(ttr_deltas, patch_artist=True, widths=0.5,
                    medianprops=dict(color="white", linewidth=2))
    for patch, v in zip(bp["boxes"], variants_to_plot):
        patch.set_facecolor(MODEL_COLORS.get(v, "#888"))
    ax.set_xticklabels([v.replace("_"," ").upper() for v in variants_to_plot], fontsize=9)
    ax.set_ylabel("|TTR(pred) − TTR(ref)|")
    ax.set_title("Lexical Diversity Gap\n(lower = more natural vocabulary diversity)")

    # 19d: Role confusion rate
    ax = axes[1, 1]
    rc_rates = []
    rc_labels = []
    for v in all_variants:
        rc = sum(1 for m in all_metrics[v] if m.role_confused) / len(all_metrics[v])
        rc_rates.append(rc * 100)
        rc_labels.append(v.replace("_"," ").upper())
    colors = [MODEL_COLORS.get(v, "#888") for v in all_variants]
    bars = ax.bar(range(len(all_variants)), rc_rates, color=colors,
                  edgecolor="white", width=0.6)
    for bar, val in zip(bars, rc_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(all_variants)))
    ax.set_xticklabels(rc_labels, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("% of turns with role confusion")
    ax.set_title("Role Confusion Rate\n(predicted user turns containing assistant language)")

    fig.tight_layout()
    _save(fig, "19_behavioural_signals.png")


def fig_judge_radar(all_metrics: dict):
    """Radar/spider chart of LLM-judge scores per model variant."""
    _set_style()
    lora_variants = sorted(v for v in all_metrics if v.startswith("lora_"))
    if not lora_variants:
        return

    # Check if judge data exists
    sample = aggregate(all_metrics[lora_variants[0]])
    if sample.get("mean_judge_overall") is None:
        print("  ⚠  No LLM-judge data — skipping radar chart.")
        return

    dims = ["Semantic\nFidelity", "Persona\nVoice", "Conversational\nCoherence",
            "Goal\nDirectedness", "Human\nRealism", "Information\nCalibration"]
    dim_keys = ["mean_judge_semantic", "mean_judge_persona", "mean_judge_coherence",
                "mean_judge_goal", "mean_judge_realism", "mean_judge_info_cal"]

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    fig.suptitle(
        "LLM-as-Judge: Turn-Level Quality Radar\n"
        "6 dimensions scored 1–5 by Claude",
        fontsize=14, fontweight="bold", y=1.02,
    )

    for variant in lora_variants:
        agg = aggregate(all_metrics[variant])
        values = [(agg.get(k) or 0) for k in dim_keys]
        values += values[:1]
        color = MODEL_COLORS.get(variant, "#888")
        ax.plot(angles, values, 'o-', linewidth=2, label=variant.replace("_"," ").upper(),
                color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=10)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    fig.tight_layout()
    _save(fig, "20_judge_radar.png")


def fig_entity_analysis(all_metrics: dict):
    """Entity-level precision/recall per model variant."""
    _set_style()
    all_variants = sorted(all_metrics.keys())
    lora_variants = [v for v in all_variants if v.startswith("lora_")]
    variants_to_plot = lora_variants if lora_variants else all_variants

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Entity-Level Factual Accuracy\n"
        "Does the model mention the right amounts, dates, merchants, card numbers?",
        fontsize=14, fontweight="bold", y=1.02,
    )

    # 21a: Entity F1 per model
    ax = axes[0]
    for v in variants_to_plot:
        non_eos = [m for m in all_metrics[v] if not m.is_eos_example and m.entity_count_ref > 0]
        if not non_eos:
            continue
        f1s = [m.entity_f1 for m in non_eos]
        ax.bar(variants_to_plot.index(v), np.mean(f1s),
               color=MODEL_COLORS.get(v, "#888"), edgecolor="white", width=0.6)
        ax.text(variants_to_plot.index(v), np.mean(f1s) + 0.02,
                f"{np.mean(f1s):.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(variants_to_plot)))
    ax.set_xticklabels([v.replace("_"," ").upper() for v in variants_to_plot], fontsize=9)
    ax.set_ylabel("Mean Entity F1")
    ax.set_ylim(0, 1.15)
    ax.set_title("Entity F1 on examples with ≥1 reference entity")

    # 21b: Precision vs Recall scatter
    ax = axes[1]
    for v in variants_to_plot:
        non_eos = [m for m in all_metrics[v] if not m.is_eos_example and m.entity_count_ref > 0]
        if not non_eos:
            continue
        prec = np.mean([m.entity_precision for m in non_eos])
        rec  = np.mean([m.entity_recall for m in non_eos])
        ax.scatter(rec, prec, s=200, color=MODEL_COLORS.get(v, "#888"),
                   label=v.replace("_"," ").upper(), zorder=3, edgecolors="white", linewidths=2)
        ax.annotate(v.replace("_"," ").upper(), (rec, prec),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax.set_xlabel("Entity Recall")
    ax.set_ylabel("Entity Precision")
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    ax.plot([0, 1], [0, 1], "--", color="#ccc", linewidth=1, alpha=0.5)
    ax.set_title("Entity Precision vs Recall\n(top-right = perfect)")
    ax.legend(fontsize=9)

    fig.tight_layout()
    _save(fig, "21_entity_analysis.png")


def fig_conversation_level(conv_rows: list):
    """Conversation-level metric distributions."""
    _set_style()
    if not conv_rows:
        return

    lora_rows = [r for r in conv_rows if r["model_variant"].startswith("lora_")]
    if not lora_rows:
        lora_rows = conv_rows
    variants = sorted(set(r["model_variant"] for r in lora_rows))

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        "Conversation-Level Metrics\n"
        "How coherent is the model across an entire conversation?",
        fontsize=14, fontweight="bold", y=1.02,
    )

    # 22a: Style compliance std per conversation
    ax = axes[0, 0]
    data = []
    for v in variants:
        vals = [r["style_compliance_std"] for r in lora_rows
                if r["model_variant"] == v and r["style_compliance_std"] is not None]
        data.append(vals)
    if any(data):
        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops=dict(color="white", linewidth=2))
        for patch, v in zip(bp["boxes"], variants):
            patch.set_facecolor(MODEL_COLORS.get(v, "#888"))
    ax.set_xticklabels([v.replace("_"," ").upper() for v in variants], fontsize=9)
    ax.set_ylabel("Style compliance std (within conversation)")
    ax.set_title("Style Coherence Across Turns\n(lower std = more consistent persona)")

    # 22b: Length trajectory correlation
    ax = axes[0, 1]
    data = []
    for v in variants:
        vals = [r["length_trajectory_correlation"] for r in lora_rows
                if r["model_variant"] == v and r["length_trajectory_correlation"] is not None]
        data.append(vals)
    if any(data):
        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops=dict(color="white", linewidth=2))
        for patch, v in zip(bp["boxes"], variants):
            patch.set_facecolor(MODEL_COLORS.get(v, "#888"))
    ax.set_xticklabels([v.replace("_"," ").upper() for v in variants], fontsize=9)
    ax.set_ylabel("Spearman ρ (pred length vs ref length)")
    ax.axhline(0, color="#C1121F", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_title("Length Trajectory Fidelity\n(higher = model length tracks reference length)")

    # 22c: Self-BLEU (mode collapse)
    ax = axes[1, 0]
    data = []
    for v in variants:
        vals = [r["self_bleu"] for r in lora_rows
                if r["model_variant"] == v and r["self_bleu"] is not None]
        data.append(vals)
    if any(data):
        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops=dict(color="white", linewidth=2))
        for patch, v in zip(bp["boxes"], variants):
            patch.set_facecolor(MODEL_COLORS.get(v, "#888"))
    ax.set_xticklabels([v.replace("_"," ").upper() for v in variants], fontsize=9)
    ax.set_ylabel("Self-BLEU (within conversation)")
    ax.set_title("Mode Collapse Detection\n(high self-BLEU = repetitive predicted turns)")

    # 22d: Conversation-level judge mean (if available)
    ax = axes[1, 1]
    has_judge = any(r.get("judge_mean") is not None for r in lora_rows)
    if has_judge:
        data = []
        for v in variants:
            vals = [r["judge_mean"] for r in lora_rows
                    if r["model_variant"] == v and r["judge_mean"] is not None]
            data.append(vals)
        if any(data):
            bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                            medianprops=dict(color="white", linewidth=2))
            for patch, v in zip(bp["boxes"], variants):
                patch.set_facecolor(MODEL_COLORS.get(v, "#888"))
        ax.set_xticklabels([v.replace("_"," ").upper() for v in variants], fontsize=9)
        ax.set_ylabel("Judge mean score (1–5)")
        ax.set_title("Conversation-Level Judge Quality")
    else:
        ax.text(0.5, 0.5, "LLM-as-judge not run\n(use --judge flag)",
                ha="center", va="center", fontsize=12, color="#999",
                transform=ax.transAxes)
        ax.set_title("Conversation-Level Judge Quality")

    fig.tight_layout()
    _save(fig, "22_conversation_level.png")


def fig_by_conversation_type(all_metrics: dict):
    """
    Figure 23: Key metrics broken out by conversation type (success / type_a / type_b).
    Shows: BLEU, hedge-rate delta, style compliance, pushback rate (type_b), prior belief
    rate (type_a). Helps diagnose whether the model behaves differently per data split.
    """
    _set_style()
    lora_variants = sorted(v for v in all_metrics if v.startswith("lora_"))
    if not lora_variants:
        return

    conv_types = ["success", "type_a", "type_b"]
    type_colors = {"success": "#4C9BE8", "type_a": "#E9C46A", "type_b": "#E76F51"}

    metrics_to_plot = [
        ("bleu_score",            "Mean BLEU",          "non_eos"),
        ("style_compliance_score","Mean Style Compliance","non_eos"),
        ("hedge_rate_delta",      "Mean Hedge-Rate Δ",  "non_eos"),
        ("is_pushback_turn",      "Pushback Turn Rate\n(type_b only)", "non_eos"),
        ("prior_belief_expressed","Prior Belief Rate\n(type_a only)",  "non_eos"),
    ]

    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 6))
    fig.suptitle(
        "Per-Conversation-Type Metrics  (LoRA variants only)\n"
        "Stratified by success / type_a (inadvertent probing) / type_b (adversarial probing)",
        fontsize=13, fontweight="bold", y=1.03,
    )

    for ax, (attr, title, pool) in zip(axes, metrics_to_plot):
        x = np.arange(len(lora_variants))
        w = 0.22
        for ci, ctype in enumerate(conv_types):
            vals = []
            for variant in lora_variants:
                subset = [
                    m for m in all_metrics[variant]
                    if m.conversation_type == ctype and
                    (pool != "non_eos" or not m.is_eos_example)
                ]
                if not subset:
                    vals.append(0.0)
                    continue
                raw = [float(getattr(m, attr)) for m in subset]
                vals.append(round(np.mean(raw), 4))
            offset = (ci - 1) * w
            bars = ax.bar(x + offset, vals, w * 0.9,
                          label=ctype, color=type_colors[ctype], alpha=0.85)
            for bar, val in zip(bars, vals):
                if val > 0.001:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01,
                            f"{val:.2f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([v.replace("_", " ").upper() for v in lora_variants],
                           fontsize=8, rotation=25, ha="right")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(ax.get_ylim()[1], 0.15))
        if ci == 0:
            ax.legend(fontsize=8, title="Conv. type")

    fig.tight_layout()
    _save(fig, "23_by_conversation_type.png")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-model prediction evaluation for UserLM")
    parser.add_argument("--judge", action="store_true",
                        help="Run LLM-as-judge evaluation (requires ANTHROPIC_API_KEY)")
    parser.add_argument("--judge-model", default=JUDGE_MODEL,
                        help=f"Model for LLM judge (default: {JUDGE_MODEL})")
    parser.add_argument("--judge-limit", type=int, default=None,
                        help="Limit number of examples to judge (for testing)")
    parser.add_argument("--no-bertscore", action="store_true",
                        help="Skip BERTScore computation")
    parser.add_argument("--no-bleurt", action="store_true",
                        help="Skip BLEURT computation")
    parser.add_argument("--bertscore-model", default=BERTSCORE_MODEL,
                        help=f"Model for BERTScore (default: {BERTSCORE_MODEL})")
    return parser.parse_args()


def main():
    args = parse_args()

    total_steps = 11
    step = 0

    def waymark(msg):
        nonlocal step
        step += 1
        print(f"\n[{step}/{total_steps}] {msg}")

    print("=" * 100)
    print("  eval_predictions.py  —  Multi-model prediction evaluation  (v2)")
    print("  Metrics: BLEU · ROUGE · METEOR · BERTScore · BLEURT · Entity F1")
    print("           Style · Hedging · Promise · TTR · Role Confusion")
    print("           LLM-as-Judge (6 dimensions)  · Conversation-level")
    print("=" * 100)

    # ── 1. Check files ───────────────────────────────────────────────────
    waymark("Checking input files …")
    missing = [name for name, path in MODEL_FILES.items() if not path.exists()]
    if missing:
        print(f"  ERROR: Missing prediction files for: {missing}")
        print("  Update the MODEL_FILES paths at the top of this script and re-run.")
        return
    for name, path in MODEL_FILES.items():
        print(f"  ✓  {name}: {path}")
    print(f"  ✓  BERTScore available: {_HAS_BERTSCORE and not args.no_bertscore}")
    print(f"  ✓  BLEURT available: {_HAS_BLEURT and not args.no_bleurt}")
    print(f"  ✓  LLM judge: {'enabled' if args.judge else 'disabled (use --judge to enable)'}")

    # ── 2. Load + verify alignment ───────────────────────────────────────
    waymark("Loading prediction files …")
    files_data = {name: load_file(path) for name, path in MODEL_FILES.items()}
    for name, rows in files_data.items():
        print(f"  {name}: {len(rows)} examples loaded")

    waymark("Verifying cross-file alignment …")
    try:
        n_examples = verify_alignment(files_data)
    except ValueError as e:
        print(f"  ERROR: {e}")
        return

    # ── 4. Compute per-example metrics (cheap: entity, TTR, hedging, etc.) ─
    waymark(f"Computing per-example metrics ({n_examples} examples × "
            f"{len(MODEL_FILES)} files × 2 variants) …")
    print("  This includes: METEOR, entity P/R/F1, TTR, hedge rate, "
          "promise rate, role confusion, style compliance")
    all_flat: list = []
    all_metrics: dict = {}

    for model_name, rows in files_data.items():
        for model_key, variant_prefix in [("base_model", "base"), ("lora_model", "lora")]:
            size_tag     = model_name.split("_")[1]
            variant_name = f"{variant_prefix}_{size_tag}"
            variant_metrics = []
            for idx, row in enumerate(rows):
                if model_key not in row:
                    continue
                m = compute_metrics_for_example(row, model_key, model_name, idx)
                variant_metrics.append(m)
                all_flat.append(m)
            all_metrics[variant_name] = variant_metrics
            n_eos = sum(1 for m in variant_metrics if m.is_eos_example)
            n_rc  = sum(1 for m in variant_metrics if m.role_confused)
            print(f"  ✓  {variant_name:<18}: {len(variant_metrics)} examples  "
                  f"({n_eos} EOS, {n_rc} role-confused)")

    print(f"  Total metric rows: {len(all_flat)}")

    # ── 5. Batch BERTScore ───────────────────────────────────────────────
    waymark("Computing BERTScore (batch) …")
    if not args.no_bertscore:
        for variant, metrics in all_metrics.items():
            non_eos = [m for m in metrics if not m.is_eos_example]
            refs  = [normalize(m.expected_output) for m in non_eos]
            preds = [normalize(m.predicted_output) for m in non_eos]
            print(f"  → {variant}: {len(non_eos)} non-EOS examples")
            P, R, F1 = compute_bertscore_batch(refs, preds, model_type=args.bertscore_model)
            for m, p, r, f in zip(non_eos, P, R, F1):
                m.bertscore_precision = p
                m.bertscore_recall    = r
                m.bertscore_f1        = f
            # EOS examples get 1.0 if correct, 0.0 if not
            for m in metrics:
                if m.is_eos_example:
                    correct = (m.predicted_output == "")
                    m.bertscore_f1 = 1.0 if correct else 0.0
                    m.bertscore_precision = m.bertscore_f1
                    m.bertscore_recall = m.bertscore_f1
    else:
        print("  ⚠  BERTScore skipped (--no-bertscore)")

    # ── 6. Batch BLEURT ──────────────────────────────────────────────────
    waymark("Computing BLEURT (batch) …")
    if not args.no_bleurt:
        for variant, metrics in all_metrics.items():
            non_eos = [m for m in metrics if not m.is_eos_example]
            refs  = [normalize(m.expected_output) for m in non_eos]
            preds = [normalize(m.predicted_output) for m in non_eos]
            print(f"  → {variant}: {len(non_eos)} non-EOS examples")
            scores = compute_bleurt_batch(refs, preds)
            for m, s in zip(non_eos, scores):
                m.bleurt_score = s
            for m in metrics:
                if m.is_eos_example:
                    correct = (m.predicted_output == "")
                    m.bleurt_score = 1.0 if correct else 0.0
    else:
        print("  ⚠  BLEURT skipped (--no-bleurt)")

    # ── 7. LLM-as-judge ─────────────────────────────────────────────────
    waymark("Running LLM-as-judge evaluation …")
    if args.judge:
        global JUDGE_MODEL
        JUDGE_MODEL = args.judge_model
        print(f"  Judge model: {JUDGE_MODEL}")
        JUDGE_CACHE_DIR.mkdir(exist_ok=True)
        judge_count = 0
        for variant, metrics in all_metrics.items():
            non_eos = [m for m in metrics if not m.is_eos_example]
            if args.judge_limit:
                non_eos = non_eos[:args.judge_limit]
            print(f"  → {variant}: judging {len(non_eos)} examples …")
            for i, m in enumerate(non_eos):
                # Find the original example data
                size_key = m.model_size
                model_key = "lora_model" if m.is_lora else "base_model"
                example = files_data[size_key][m.example_idx]
                persona = example.get("_meta", {}).get("persona", {})

                result = run_llm_judge(
                    example, m.predicted_output, m.expected_output,
                    persona, variant, m.example_idx,
                )
                m.judge_semantic_fidelity = result.get("semantic_fidelity")
                m.judge_persona_voice = result.get("persona_voice")
                m.judge_conversational_coherence = result.get("conversational_coherence")
                m.judge_goal_directedness = result.get("goal_directedness")
                m.judge_human_realism = result.get("human_realism")
                m.judge_information_calibration = result.get("information_calibration")

                scores = [v for v in [
                    m.judge_semantic_fidelity, m.judge_persona_voice,
                    m.judge_conversational_coherence, m.judge_goal_directedness,
                    m.judge_human_realism, m.judge_information_calibration,
                ] if v is not None]
                m.judge_mean = round(np.mean(scores), 2) if scores else None

                judge_count += 1
                if judge_count % 10 == 0:
                    print(f"    … {judge_count} examples judged")
                time.sleep(0.2)  # rate limit courtesy

        print(f"  ✓  {judge_count} total examples judged")
    else:
        print("  ⚠  LLM judge skipped (use --judge to enable)")

    # ── 8. Cross-model comparison ────────────────────────────────────────
    waymark("Running cross-model comparison (Wilcoxon signed-rank tests) …")
    comparison_rows = compare_models(all_metrics)
    print(f"  ✓  {len(comparison_rows)} pairwise comparisons computed")

    # ── 9. Conversation-level metrics ────────────────────────────────────
    waymark("Computing conversation-level metrics (grouped by generation_idx) …")
    conv_rows = compute_conversation_metrics(all_metrics)
    n_convs = len(set((r["model_variant"], r["generation_idx"]) for r in conv_rows))
    print(f"  ✓  {len(conv_rows)} conversation × variant rows  "
          f"({n_convs} unique conversations)")

    # ── 10. Print summary + write CSVs ───────────────────────────────────
    print_summary(all_metrics)

    waymark("Writing CSVs …")
    write_detailed_csv(all_flat, DETAILED_CSV)
    print(f"  → {DETAILED_CSV.name}")
    write_category_csv(all_metrics, CATEGORY_CSV)
    print(f"  → {CATEGORY_CSV.name}")
    write_comparison_csv(comparison_rows, COMPARISON_CSV)
    print(f"  → {COMPARISON_CSV.name}")
    write_conversation_csv(conv_rows, CONVERSATION_CSV)
    print(f"  → {CONVERSATION_CSV.name}")

    # ── 11. Generate figures ─────────────────────────────────────────────
    waymark("Generating visualisations …")
    IMAGES_DIR.mkdir(exist_ok=True)

    print("  [a] Category heatmap …")
    fig_category_heatmap(all_metrics)

    print("  [b] Model comparison …")
    fig_model_comparison(all_metrics, comparison_rows)

    print("  [c] EOS analysis …")
    fig_eos_analysis(all_metrics)

    print("  [d] Efficiency frontier …")
    fig_efficiency_frontier(all_metrics)

    print("  [e] Style compliance …")
    fig_style_compliance(all_metrics)

    print("  [f] BERTScore comparison …")
    fig_bertscore_comparison(all_metrics)

    print("  [g] Behavioural signals …")
    fig_behavioural_signals(all_metrics)

    print("  [h] Entity analysis …")
    fig_entity_analysis(all_metrics)

    print("  [i] LLM-judge radar …")
    fig_judge_radar(all_metrics)

    print("  [j] Conversation-level metrics …")
    fig_conversation_level(conv_rows)

    print("  [k] By conversation type (success / type_a / type_b) …")
    fig_by_conversation_type(all_metrics)

    print("  ✓  All 11 figures saved.")

    print("\n" + "=" * 100)
    print("  Done. Outputs:")
    print(f"    CSVs:    {DETAILED_CSV.name}, {CATEGORY_CSV.name}, "
          f"{COMPARISON_CSV.name}, {CONVERSATION_CSV.name}")
    print(f"    Images:  {IMAGES_DIR}/13_*.png … 23_*.png")
    if args.judge:
        print(f"    Cache:   {JUDGE_CACHE_DIR}/")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
