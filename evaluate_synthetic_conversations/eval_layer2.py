#!/usr/bin/env python3
"""
eval_layer2.py
LLM-as-judge evaluation of synthetic banking dispute conversations.

Scores each conversation on 5 behavioral dimensions grounded in:
  - Wang et al. (2025)  — depth-first questioning, uncertainty expression
  - Mannekote et al.    — pragmatic naturalness
  - Chen et al. (2024)  — persona fidelity / diversity

Scores are 1–5 where 1 = LLM-like behaviour, 5 = human-like behaviour.

Usage:
    # Evaluate all 80 conversations (uses ANTHROPIC_API_KEY env var)
    python eval_layer2.py

    # Override API key and model
    python eval_layer2.py --api-key sk-ant-... --model claude-haiku-4-5-20251001

    # Dry run on 5 conversations to test prompt quality
    python eval_layer2.py --limit 5

    # Re-evaluate everything, ignoring cached results
    python eval_layer2.py --force

Outputs:
    - layer2_cache/conv_NNN.json   — per-conversation judge response (cached)
    - eval_results_layer2.csv      — all scores, one row per conversation
    - images/08_*.png … 12_*.png   — professional visualisation suite
"""

import os
import re
import csv
import json
import time
import math
import argparse
import random
from pathlib import Path
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from typing import Optional

import anthropic

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np

# ─────────────────────────────────────────────
#  Config — all tuneable constants live here
# ─────────────────────────────────────────────

DUMPS_DIR   = Path(__file__).parent / "conversation_dumps"
CACHE_DIR   = Path(__file__).parent / "layer2_cache"
OUTPUT_CSV  = Path(__file__).parent / "eval_results_layer2.csv"
L1_CSV      = Path(__file__).parent / "eval_results.csv"       # for Fig 10 correlation
IMAGES_DIR  = Path(__file__).parent / "images"

DEFAULT_MODEL = "claude-sonnet-4-6"

# Dimension weights for the composite overall_score (must sum to 1.0).
# Rationale:
#   depth_first  + pragmatic  weighted highest — Wang et al.'s most actionable findings
#   uncertainty  medium       — strong LLM signal, partially overlaps Layer 1
#   info_drip    lowest       — mostly captured by Layer 1 info density
#   persona      lowest       — partially captured by Layer 1 persona checks
DIMENSION_WEIGHTS = {
    "depth_first": 0.25,
    "uncertainty": 0.20,
    "info_drip":   0.15,
    "pragmatic":   0.25,
    "persona":     0.15,
}

# Retry config for API calls
MAX_RETRIES    = 4
RETRY_BASE_SEC = 2.0    # exponential backoff base
CALL_DELAY_SEC = 0.5    # polite sleep between calls

# ─────────────────────────────────────────────
#  Shared visual style (mirrors eval_conversations.py)
# ─────────────────────────────────────────────

PALETTE = {
    "calm":              "#4C9BE8",
    "mildly_frustrated": "#F4A261",
    "very_upset":        "#E76F51",
    "escalating":        "#C1121F",
    "direct":            "#2A9D8F",
    "indirect":          "#E9C46A",
    "terse":             "#A8DADC",
    "verbose":           "#F4A261",
    "novice":            "#BDE0FE",
    "intermediate":      "#6A8EAE",
    "expert":            "#1B3A4B",
    "clear":             "#2A9D8F",
    "vague":             "#E9C46A",
    "wrong_mental_model":"#E76F51",
    # dimension colours
    "depth_first": "#457B9D",
    "uncertainty": "#2A9D8F",
    "info_drip":   "#E9C46A",
    "pragmatic":   "#A8DADC",
    "persona":     "#F4A261",
    "overall":     "#1B3A4B",
}

DIM_LABELS = {
    "depth_first": "Depth-First\nQuestioning",
    "uncertainty": "Appropriate\nUncertainty",
    "info_drip":   "Natural\nInfo Drip",
    "pragmatic":   "Pragmatic\nNaturalness",
    "persona":     "Persona\nFidelity",
    "overall":     "Overall",
}

DIMENSIONS = ["depth_first", "uncertainty", "info_drip", "pragmatic", "persona"]

def _set_style():
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "font.family":      "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.titlesize":   13,
        "axes.labelsize":   11,
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
    })

# ─────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────

@dataclass
class Turn:
    speaker: str
    text: str

@dataclass
class Conversation:
    idx: int
    knowledge_level: str     = ""
    emotional_state: str     = ""
    communication_style: str = ""
    goal_clarity: str        = ""
    certainty: str           = ""
    information_completeness: str = ""
    prior_contact: str       = ""
    expected_resolution: str = ""
    merchant: str            = ""
    amount: str              = ""
    transaction_date: str    = ""
    turns: list = field(default_factory=list)

@dataclass
class Layer2Result:
    conv_idx: int
    # Persona metadata (copied for convenience)
    knowledge_level: str     = ""
    emotional_state: str     = ""
    communication_style: str = ""
    goal_clarity: str        = ""
    certainty: str           = ""
    # Dimension scores (1–5)
    depth_first_score: float  = 0.0
    uncertainty_score: float  = 0.0
    info_drip_score:   float  = 0.0
    pragmatic_score:   float  = 0.0
    persona_score:     float  = 0.0
    overall_score:     float  = 0.0
    # Qualitative fields from judge
    depth_first_rationale: str  = ""
    uncertainty_rationale: str  = ""
    info_drip_rationale:   str  = ""
    pragmatic_rationale:   str  = ""
    persona_rationale:     str  = ""
    standout_issue:   str       = ""
    standout_quality: str       = ""
    # Metadata
    model_used: str  = ""
    parse_error: str = ""

# ─────────────────────────────────────────────
#  Parser (identical logic to eval_conversations.py)
# ─────────────────────────────────────────────

def parse_conversation(filepath: Path) -> Optional[Conversation]:
    text = filepath.read_text(encoding="utf-8")
    lines = text.splitlines()

    idx = None
    for line in lines:
        m = re.match(r"CONVERSATION\s+(\d+)", line.strip())
        if m:
            idx = int(m.group(1))
            break
    if idx is None:
        return None

    conv = Conversation(idx=idx)
    in_persona = in_scenario = in_dialogue = False

    for line in lines:
        stripped = line.strip()

        if stripped == "PERSONA:":
            in_persona = True;  in_scenario = False;  continue
        if stripped == "SCENARIO:":
            in_scenario = True; in_persona  = False;  continue
        if re.match(r"-{4,}", stripped):
            in_persona = False; in_scenario = False;  continue

        kv = re.match(r"\s+([\w_]+)\s*:\s*(.+)", line)
        if kv:
            key   = kv.group(1).strip().lower()
            value = kv.group(2).strip()
            if in_persona:
                if key == "knowledge_level":       conv.knowledge_level       = value
                elif key == "emotional_state":     conv.emotional_state       = value
                elif key == "communication_style": conv.communication_style   = value
                elif key == "goal_clarity":        conv.goal_clarity          = value
            elif in_scenario:
                if key == "certainty":               conv.certainty               = value
                elif key == "information_completeness": conv.information_completeness = value
                elif key == "prior_contact":          conv.prior_contact            = value
                elif key == "expected_resolution":    conv.expected_resolution      = value
                elif key == "merchant":               conv.merchant                 = value
                elif key == "amount":                 conv.amount                   = value
                elif key == "transaction_date":       conv.transaction_date         = value
            continue

        m = re.match(r"^(Customer|Agent):\s*(.+)", stripped)
        if m:
            in_dialogue = True
            in_persona  = False
            in_scenario = False
            conv.turns.append(Turn(speaker=m.group(1), text=m.group(2).strip()))
            continue

        if in_dialogue and conv.turns and stripped and not re.match(r"=+|-{4,}", stripped):
            if not re.match(r"^(Customer|Agent):", stripped):
                conv.turns[-1].text += " " + stripped

    return conv

# ─────────────────────────────────────────────
#  Judge prompt construction
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert dialogue quality evaluator specialising in task-oriented \
conversational AI for the banking and financial services domain.

Your role is to evaluate whether the CUSTOMER side of a banking dispute conversation \
reads like a real human caller or like an LLM prompted to play a user. \
You are grounded in empirical findings from dialogue research:

- Wang et al. (2025): Real humans use depth-first questioning (fully exhaust one concern \
before raising the next). LLM-generated users use breadth-first questioning (bundle multiple \
concerns in one turn). Real humans are tentative and uncertain; LLMs over-commit with phrases \
like "definitely", "absolutely", "certainly".

- Mannekote et al. (2025): Naturalness is pragmatic alignment, not just surface fluency. \
A natural customer request fits the conversational context — it is appropriately indirect \
or direct, and never sounds like a form or script.

- Chen et al. (2024): Diverse personas must actually manifest in the dialogue. A "novice" \
customer should speak differently from an "expert". A "calm" tone must differ from \
"mildly_frustrated".

You evaluate ONLY the customer turns. Ignore whether the agent is good or bad."""


def build_user_prompt(conv: Conversation) -> str:
    # Format the conversation as a readable transcript
    transcript_lines = []
    for turn in conv.turns:
        prefix = "CUSTOMER" if turn.speaker == "Customer" else "AGENT"
        transcript_lines.append(f"[{prefix}]: {turn.text}")
    transcript = "\n".join(transcript_lines)

    return f"""Below is a synthetic banking dispute conversation with its persona and scenario metadata.
Evaluate ONLY the CUSTOMER turns on the five dimensions below.

━━━━━━━━━━━━━━━━━━━━━━━━
PERSONA ASSIGNED TO CUSTOMER
━━━━━━━━━━━━━━━━━━━━━━━━
knowledge_level:     {conv.knowledge_level}
emotional_state:     {conv.emotional_state}
communication_style: {conv.communication_style}
goal_clarity:        {conv.goal_clarity}

SCENARIO
━━━━━━━━━━━━━━━━━━━━━━━━
certainty:                {conv.certainty}
information_completeness: {conv.information_completeness}
prior_contact:            {conv.prior_contact}
expected_resolution:      {conv.expected_resolution}
merchant:                 {conv.merchant}
amount:                   {conv.amount}
transaction_date:         {conv.transaction_date}

━━━━━━━━━━━━━━━━━━━━━━━━
TRANSCRIPT
━━━━━━━━━━━━━━━━━━━━━━━━
{transcript}

━━━━━━━━━━━━━━━━━━━━━━━━
SCORING RUBRIC
━━━━━━━━━━━━━━━━━━━━━━━━

Score each dimension 1–5:
  1 = clearly LLM-like behaviour
  3 = mixed / borderline
  5 = clearly human-like behaviour

DIMENSION 1 — DEPTH-FIRST QUESTIONING
  5: Customer fully resolves one concern before raising another. Never bundles.
  3: One or two instances of bundling concerns, but mostly sequential.
  1: Customer front-loads multiple concerns (dispute + card cancel + refund ETA) \
in the same message, especially early turns.

DIMENSION 2 — APPROPRIATE UNCERTAINTY EXPRESSION
  5: Uncertainty language perfectly matches the scenario. Uncertain scenario \
("certainty: uncertain") → customer hedges ("I think", "not sure", "could be"). \
Certain fraud scenario → customer can be assertive, but not robotic.
  3: Mostly appropriate but with some mismatched certainty phrases.
  1: Uncertain scenario but customer says "I definitely did not make this charge" \
or uses other commitment language throughout.

DIMENSION 3 — NATURAL INFORMATION DRIP
  5: Customer reveals {conv.merchant} / {conv.amount} / {conv.transaction_date} \
gradually, only when prompted or naturally relevant. Does not front-load all facts.
  3: Mixed — some facts offered proactively but some held back.
  1: Customer volunteers all key facts (merchant + amount + date) unprompted in \
the first message.

DIMENSION 4 — PRAGMATIC NATURALNESS
  5: Every customer turn sounds like something a real caller would say. \
Phrasing is colloquial, appropriately informal, and fits the conversational context. \
No turns read like a template or a script.
  3: Most turns natural but 1–2 turns feel formulaic or overly formal.
  1: Multiple turns sound scripted or robotic — phrasing a real person would never use.

DIMENSION 5 — PERSONA FIDELITY
  5: The assigned persona strongly manifests. A "{conv.emotional_state}" tone is \
clearly present and sustained. A "{conv.knowledge_level}" knowledge level shows in \
vocabulary and question choice. A "{conv.communication_style}" style is apparent \
in turn structure and length.
  3: Persona partially shows — some attributes manifest but others are absent or \
contradict the label.
  1: The assigned persona is not detectable. Conversation could have any persona label.

━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━

Return ONLY a valid JSON object with exactly these keys. No markdown, no explanation outside the JSON.

{{
  "depth_first_score": <integer 1-5>,
  "depth_first_rationale": "<one sentence citing a specific turn>",
  "uncertainty_score": <integer 1-5>,
  "uncertainty_rationale": "<one sentence citing a specific turn>",
  "info_drip_score": <integer 1-5>,
  "info_drip_rationale": "<one sentence citing a specific turn>",
  "pragmatic_score": <integer 1-5>,
  "pragmatic_rationale": "<one sentence citing a specific turn>",
  "persona_score": <integer 1-5>,
  "persona_rationale": "<one sentence citing a specific turn>",
  "standout_issue": "<the single most human-unrealistic thing in this conversation, or empty string>",
  "standout_quality": "<the single most realistic thing, or empty string>"
}}"""

# ─────────────────────────────────────────────
#  API call with retry + exponential backoff
# ─────────────────────────────────────────────

def call_judge(client: anthropic.Anthropic, conv: Conversation,
               model: str) -> dict:
    """Call the LLM judge and return the parsed JSON dict."""
    user_prompt = build_user_prompt(conv)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=512,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = response.content[0].text.strip()

            # Strip markdown code fences if model wraps the JSON
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$",           "", raw)

            return json.loads(raw)

        except anthropic.RateLimitError:
            wait = RETRY_BASE_SEC * (2 ** attempt) + random.uniform(0, 1)
            print(f"    ⚠  Rate limit — waiting {wait:.1f}s (attempt {attempt+1}/{MAX_RETRIES})")
            time.sleep(wait)

        except anthropic.APIError as e:
            wait = RETRY_BASE_SEC * (2 ** attempt)
            print(f"    ⚠  API error: {e} — waiting {wait:.1f}s")
            time.sleep(wait)

        except json.JSONDecodeError as e:
            # Return a parse error dict so we can still log the failure
            return {"_parse_error": str(e), "_raw": raw if "raw" in dir() else ""}

    return {"_parse_error": f"Max retries ({MAX_RETRIES}) exceeded"}

# ─────────────────────────────────────────────
#  Cache helpers
# ─────────────────────────────────────────────

def cache_path(conv_idx: int) -> Path:
    return CACHE_DIR / f"conv_{conv_idx:03d}.json"

def load_cache(conv_idx: int) -> Optional[dict]:
    p = cache_path(conv_idx)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    return None

def save_cache(conv_idx: int, data: dict):
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path(conv_idx).write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

# ─────────────────────────────────────────────
#  Score computation
# ─────────────────────────────────────────────

def compute_result(conv: Conversation, judge: dict, model: str) -> Layer2Result:
    r = Layer2Result(
        conv_idx            = conv.idx,
        knowledge_level     = conv.knowledge_level,
        emotional_state     = conv.emotional_state,
        communication_style = conv.communication_style,
        goal_clarity        = conv.goal_clarity,
        certainty           = conv.certainty,
        model_used          = model,
    )

    if "_parse_error" in judge:
        r.parse_error = judge["_parse_error"]
        return r

    def _get_score(key: str) -> float:
        raw = judge.get(key, 0)
        try:
            return max(1.0, min(5.0, float(raw)))
        except (TypeError, ValueError):
            return 0.0

    r.depth_first_score = _get_score("depth_first_score")
    r.uncertainty_score = _get_score("uncertainty_score")
    r.info_drip_score   = _get_score("info_drip_score")
    r.pragmatic_score   = _get_score("pragmatic_score")
    r.persona_score     = _get_score("persona_score")

    # Weighted overall score
    r.overall_score = round(
        r.depth_first_score * DIMENSION_WEIGHTS["depth_first"] +
        r.uncertainty_score * DIMENSION_WEIGHTS["uncertainty"] +
        r.info_drip_score   * DIMENSION_WEIGHTS["info_drip"]   +
        r.pragmatic_score   * DIMENSION_WEIGHTS["pragmatic"]   +
        r.persona_score     * DIMENSION_WEIGHTS["persona"],
        3,
    )

    r.depth_first_rationale = judge.get("depth_first_rationale", "")
    r.uncertainty_rationale = judge.get("uncertainty_rationale", "")
    r.info_drip_rationale   = judge.get("info_drip_rationale",   "")
    r.pragmatic_rationale   = judge.get("pragmatic_rationale",   "")
    r.persona_rationale     = judge.get("persona_rationale",     "")
    r.standout_issue        = judge.get("standout_issue",        "")
    r.standout_quality      = judge.get("standout_quality",      "")

    return r

# ─────────────────────────────────────────────
#  CSV output
# ─────────────────────────────────────────────

def write_csv(results: list, path: Path):
    fieldnames = [
        "conv_idx", "knowledge_level", "emotional_state",
        "communication_style", "goal_clarity", "certainty",
        "depth_first_score", "uncertainty_score", "info_drip_score",
        "pragmatic_score", "persona_score", "overall_score",
        "depth_first_rationale", "uncertainty_rationale", "info_drip_rationale",
        "pragmatic_rationale", "persona_rationale",
        "standout_issue", "standout_quality",
        "model_used", "parse_error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = asdict(r)
            writer.writerow({k: row[k] for k in fieldnames})
    print(f"\n✓  Results written to: {path}")

# ─────────────────────────────────────────────
#  Load Layer 1 results for correlation plots
# ─────────────────────────────────────────────

def load_layer1() -> dict:
    """Returns dict keyed by conv_idx (int) → row dict."""
    if not L1_CSV.exists():
        return {}
    with open(L1_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {int(row["conv_idx"]): row for row in reader}

# ─────────────────────────────────────────────
#  Terminal summary
# ─────────────────────────────────────────────

def print_summary(results: list):
    valid = [r for r in results if not r.parse_error]
    n = len(valid)
    if n == 0:
        print("No valid results to summarise.")
        return

    sep = "─" * 70
    print(f"\n{'=' * 70}")
    print(f"  LAYER 2 EVALUATION SUMMARY  ({n} conversations scored)")
    print(f"{'=' * 70}")

    print(f"\n{sep}")
    print("  DIMENSION SCORES  (1 = LLM-like  ←—→  5 = Human-like)")
    print(sep)
    print(f"  {'Dimension':<30} {'Mean':>6}  {'Min':>5}  {'Max':>5}  {'Stdev':>6}")
    print(f"  {'─'*30}  {'─'*6}  {'─'*5}  {'─'*5}  {'─'*6}")

    all_dims = DIMENSIONS + ["overall"]
    for dim in all_dims:
        attr = f"{dim}_score" if dim != "overall" else "overall_score"
        vals = [getattr(r, attr) for r in valid if getattr(r, attr) > 0]
        if vals:
            print(f"  {DIM_LABELS[dim].replace(chr(10),' '):<30} "
                  f"{np.mean(vals):>6.2f}  {min(vals):>5.1f}  {max(vals):>5.1f}  "
                  f"{np.std(vals):>6.2f}")

    # Bottom 10 by overall score
    bottom = sorted(valid, key=lambda r: r.overall_score)[:10]
    print(f"\n{sep}")
    print("  LOWEST-SCORING CONVERSATIONS (priority regeneration candidates)")
    print(sep)
    for r in bottom:
        print(f"  Conv {r.conv_idx:03d}  overall={r.overall_score:.2f}  "
              f"[d={r.depth_first_score:.0f} u={r.uncertainty_score:.0f} "
              f"i={r.info_drip_score:.0f} p={r.pragmatic_score:.0f} "
              f"pe={r.persona_score:.0f}]  "
              f"({r.emotional_state}, {r.knowledge_level})")
        if r.standout_issue:
            print(f"         ↳ {r.standout_issue}")

    parse_errors = [r for r in results if r.parse_error]
    if parse_errors:
        print(f"\n  ⚠  {len(parse_errors)} conversation(s) had parse errors: "
              f"{[r.conv_idx for r in parse_errors]}")

    print(f"{'=' * 70}\n")

# ─────────────────────────────────────────────
#  Visualisations
# ─────────────────────────────────────────────

def _save(fig, name: str):
    IMAGES_DIR.mkdir(exist_ok=True)
    path = IMAGES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {path.name}")


# ── Figure 8 — Score Distributions ───────────────────────────────────────────
def fig_score_distributions(results: list):
    """
    Violin + strip plot for each of the 5 dimensions and overall score.
    Immediately shows which dimensions score poorly across the dataset.
    """
    _set_style()
    valid = [r for r in results if not r.parse_error]

    fig, axes = plt.subplots(1, 6, figsize=(17, 6), sharey=True)
    fig.suptitle(
        "Layer 2 — Score Distributions per Dimension\n"
        "1 = LLM-like behaviour   ·   5 = Human-like behaviour   "
        f"(n={len(valid)} conversations, model: {valid[0].model_used if valid else '?'})",
        fontsize=13, fontweight="bold", y=1.02,
    )

    all_dims = DIMENSIONS + ["overall"]

    for ax, dim in zip(axes, all_dims):
        attr  = f"{dim}_score" if dim != "overall" else "overall_score"
        vals  = [getattr(r, attr) for r in valid if getattr(r, attr) > 0]
        color = PALETTE.get(dim, "#888888")

        # Violin
        parts = ax.violinplot(vals, positions=[0], showmedians=False,
                              showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.55)
            pc.set_edgecolor("#333333")
            pc.set_linewidth(0.8)

        # Box overlay
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.vlines(0, q1, q3, color="#333333", linewidth=5, alpha=0.6, zorder=3)
        ax.scatter(0, med, color="white", s=30, zorder=4, edgecolors="#333333",
                   linewidths=1.2)

        # Jitter strip
        jitter = np.random.uniform(-0.12, 0.12, len(vals))
        ax.scatter(jitter, vals, color=color, alpha=0.5, s=18, zorder=2,
                   edgecolors="white", linewidths=0.4)

        mean_val = np.mean(vals)
        ax.axhline(mean_val, color="#C1121F", linewidth=1.2, linestyle="--", alpha=0.7)
        ax.text(0.5, mean_val + 0.08, f"μ={mean_val:.2f}",
                transform=ax.get_yaxis_transform(), ha="right",
                fontsize=8.5, color="#C1121F", fontweight="bold")

        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(0.5, 5.5)
        ax.set_xticks([])
        ax.set_title(DIM_LABELS[dim], fontsize=10, pad=6)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
        if ax == axes[0]:
            ax.set_ylabel("Score (1–5)", fontsize=10)

    # Shared reference lines annotation
    axes[-1].annotate("— mean", xy=(1, 0), xycoords="axes fraction",
                      fontsize=8, color="#C1121F", ha="right", va="bottom")

    # Weight labels below each dimension
    for ax, dim in zip(axes[:-1], DIMENSIONS):
        ax.text(0.5, -0.10, f"w={DIMENSION_WEIGHTS[dim]:.2f}",
                transform=ax.transAxes, ha="center", fontsize=8.5, color="#666666")

    fig.tight_layout()
    _save(fig, "08_score_distributions.png")


# ── Figure 9 — Scores by Persona Attribute ───────────────────────────────────
def fig_scores_by_persona(results: list):
    """
    Four heatmaps (one per persona attribute) showing mean score per dimension.
    Reveals which persona types are hardest to generate realistically.
    """
    _set_style()
    valid = [r for r in results if not r.parse_error]

    persona_attrs = [
        ("emotional_state",     "Emotional State"),
        ("knowledge_level",     "Knowledge Level"),
        ("communication_style", "Communication Style"),
        ("goal_clarity",        "Goal Clarity"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        "Layer 2 — Mean Scores by Persona Attribute\n"
        "Which persona combinations are hardest to generate realistically?",
        fontsize=14, fontweight="bold", y=1.02,
    )

    dim_labels_short = [DIM_LABELS[d].replace("\n", " ") for d in DIMENSIONS] + ["Overall"]
    all_dims_with_overall = DIMENSIONS + ["overall"]

    for ax, (attr, attr_label) in zip(axes.flat, persona_attrs):
        attr_values = sorted(set(getattr(r, attr) for r in valid))

        matrix = []
        for av in attr_values:
            row = []
            subset = [r for r in valid if getattr(r, attr) == av]
            for dim in all_dims_with_overall:
                score_attr = f"{dim}_score" if dim != "overall" else "overall_score"
                vals = [getattr(r, score_attr) for r in subset if getattr(r, score_attr) > 0]
                row.append(round(np.mean(vals), 2) if vals else 0.0)
            matrix.append(row)

        matrix_np = np.array(matrix)

        im = ax.imshow(matrix_np, vmin=1, vmax=5, cmap="RdYlGn", aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03,
                     label="Mean score (1=LLM-like, 5=human-like)")

        ax.set_xticks(range(len(all_dims_with_overall)))
        ax.set_xticklabels(dim_labels_short, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(attr_values)))
        ax.set_yticklabels([av.replace("_", " ") for av in attr_values], fontsize=10)
        ax.set_title(f"By {attr_label}", pad=8)

        # Annotate each cell with the mean value
        for i in range(len(attr_values)):
            for j in range(len(all_dims_with_overall)):
                val = matrix_np[i, j]
                text_color = "black" if 2.0 < val < 4.2 else "white"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=text_color)

    fig.tight_layout()
    _save(fig, "09_scores_by_persona.png")


# ── Figure 10 — Layer 1 vs Layer 2 Correlation ───────────────────────────────
def fig_layer1_vs_layer2(results: list, l1_data: dict):
    """
    Four scatter plots validating whether Layer 1 heuristics predicted
    the LLM judge's scores. If they don't correlate, Layer 1 was measuring
    something different from what the judge penalises.
    """
    _set_style()
    valid = [r for r in results if not r.parse_error and r.conv_idx in l1_data]

    if len(valid) < 5:
        print("  ⚠  Not enough Layer 1 data to produce Fig 10 — skipping.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        "Layer 1 Heuristics vs Layer 2 LLM-Judge Scores\n"
        "Validates whether rule-based signals predict behavioural judgment",
        fontsize=14, fontweight="bold", y=1.02,
    )

    def _scatter(ax, l1_key, l2_attr, xlabel, ylabel, title, jitter=False):
        x_vals, y_vals, colors = [], [], []
        for r in valid:
            l1 = l1_data[r.conv_idx]
            try:
                x = float(l1[l1_key])
            except (KeyError, ValueError):
                continue
            y = getattr(r, l2_attr)
            if y == 0:
                continue
            if jitter:
                x += random.uniform(-0.08, 0.08)
            x_vals.append(x)
            y_vals.append(y)
            colors.append(PALETTE.get(r.emotional_state, "#888888"))

        ax.scatter(x_vals, y_vals, c=colors, alpha=0.7, s=55,
                   edgecolors="white", linewidths=0.5)

        # Correlation line
        if len(x_vals) > 3:
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            xs = np.linspace(min(x_vals), max(x_vals), 100)
            ax.plot(xs, p(xs), color="#C1121F", linewidth=1.4, linestyle="--",
                    alpha=0.8)
            corr = np.corrcoef(x_vals, y_vals)[0, 1]
            ax.text(0.97, 0.05, f"r = {corr:.2f}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=10, fontweight="bold",
                    color="#C1121F",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="#C1121F", alpha=0.8))

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
        ax.set_ylim(0.5, 5.5)

    # Emotional state legend
    states = sorted(set(r.emotional_state for r in valid))
    patches = [mpatches.Patch(color=PALETTE.get(s, "#888"), label=s.replace("_", " "))
               for s in states]

    _scatter(axes[0, 0],
             l1_key="hedge_minus_certainty",
             l2_attr="uncertainty_score",
             xlabel="Layer 1: Hedge − Certainty rate",
             ylabel="Layer 2: Uncertainty score (1–5)",
             title="Hedging signal → Uncertainty judgment")

    _scatter(axes[0, 1],
             l1_key="info_density_score",
             l2_attr="info_drip_score",
             xlabel="Layer 1: Info density in turn 1 (0–1)",
             ylabel="Layer 2: Info drip score (1–5)",
             title="Info density → Info drip judgment")

    _scatter(axes[1, 0],
             l1_key="breadth_first_flag",
             l2_attr="depth_first_score",
             xlabel="Layer 1: Breadth-first flag (0 or 1)",
             ylabel="Layer 2: Depth-first score (1–5)",
             title="Breadth-first flag → Depth-first judgment",
             jitter=True)

    _scatter(axes[1, 1],
             l1_key="words_mean",
             l2_attr="pragmatic_score",
             xlabel="Layer 1: Avg words per customer turn",
             ylabel="Layer 2: Pragmatic naturalness (1–5)",
             title="Turn length → Pragmatic naturalness")

    # Shared legend
    fig.legend(handles=patches, loc="lower center", ncol=len(patches),
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.03))

    fig.tight_layout()
    _save(fig, "10_layer1_vs_layer2_correlation.png")


# ── Figure 11 — Bottom Conversations ─────────────────────────────────────────
def fig_bottom_conversations(results: list, n_bottom: int = 15):
    """
    Horizontal stacked bar showing each low-scoring conversation's dimension
    breakdown — the regeneration priority list.
    """
    _set_style()
    valid = sorted(
        [r for r in results if not r.parse_error and r.overall_score > 0],
        key=lambda r: r.overall_score
    )[:n_bottom]

    if not valid:
        print("  ⚠  No valid results for Fig 11 — skipping.")
        return

    fig, ax = plt.subplots(figsize=(13, max(5, len(valid) * 0.55 + 2)))
    fig.suptitle(
        f"Bottom {len(valid)} Conversations by Overall Score\n"
        "Priority candidates for manual review or regeneration before scaling up",
        fontsize=13, fontweight="bold", y=1.02,
    )

    y_labels = []
    for r in valid:
        persona_short = (f"Conv {r.conv_idx:03d}  "
                         f"{r.emotional_state.replace('_',' ')} · "
                         f"{r.knowledge_level} · {r.communication_style}\n"
                         f"overall={r.overall_score:.2f}")
        y_labels.append(persona_short)

    y_pos = range(len(valid))
    left = np.zeros(len(valid))

    for dim in DIMENSIONS:
        scores = [getattr(r, f"{dim}_score") for r in valid]
        bars = ax.barh(list(y_pos), scores, left=left,
                       color=PALETTE[dim], edgecolor="white",
                       linewidth=0.6, height=0.65, label=DIM_LABELS[dim].replace("\n", " "))
        # Label each segment if wide enough
        for bar, val, l in zip(bars, scores, left):
            if val > 0.4:
                ax.text(l + val / 2, bar.get_y() + bar.get_height() / 2,
                        f"{val:.0f}", va="center", ha="center",
                        fontsize=8, color="black", fontweight="bold")
        left += np.array(scores, dtype=float)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(y_labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Cumulative score across 5 dimensions (max = 25)")
    ax.axvline(np.mean([r.overall_score for r in valid]) * 5,
               color="#C1121F", linewidth=1.4, linestyle="--", alpha=0.7,
               label="Avg overall (this group)")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    _save(fig, "11_bottom_conversations.png")


# ── Figure 12 — Radar Overview ────────────────────────────────────────────────
def fig_radar_overview(results: list):
    """
    Radar / spider chart showing mean score per dimension, with separate
    lines for each emotional_state group. Single-glance quality card.
    """
    _set_style()
    valid = [r for r in results if not r.parse_error]

    dims_for_radar = DIMENSIONS
    labels = [DIM_LABELS[d].replace("\n", " ") for d in dims_for_radar]
    N = len(dims_for_radar)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]   # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8),
                           subplot_kw=dict(polar=True))
    fig.suptitle(
        "Dataset Quality Radar — Mean Score per Dimension\n"
        "Outer edge = 5 (human-like)   ·   Inner = 1 (LLM-like)\n"
        "Broken down by emotional state",
        fontsize=13, fontweight="bold", y=1.04,
    )

    # Overall mean line
    overall_means = []
    for dim in dims_for_radar:
        attr = f"{dim}_score"
        vals = [getattr(r, attr) for r in valid if getattr(r, attr) > 0]
        overall_means.append(np.mean(vals) if vals else 0)
    overall_means += overall_means[:1]

    ax.plot(angles, overall_means, color="#1B3A4B", linewidth=2.5,
            linestyle="-", label="All conversations", zorder=5)
    ax.fill(angles, overall_means, color="#1B3A4B", alpha=0.12)

    # Per emotional-state lines
    states = sorted(set(r.emotional_state for r in valid))
    for state in states:
        subset = [r for r in valid if r.emotional_state == state]
        means = []
        for dim in dims_for_radar:
            attr = f"{dim}_score"
            vals = [getattr(r, attr) for r in subset if getattr(r, attr) > 0]
            means.append(np.mean(vals) if vals else 0)
        means += means[:1]
        color = PALETTE.get(state, "#888888")
        ax.plot(angles, means, color=color, linewidth=1.6, linestyle="--",
                alpha=0.8, label=state.replace("_", " "))
        ax.fill(angles, means, color=color, alpha=0.05)

    # Reference circles
    for level in [1, 2, 3, 4, 5]:
        ax.plot(angles, [level] * (N + 1), color="#cccccc", linewidth=0.6,
                linestyle=":", zorder=0)
        ax.text(angles[0], level + 0.1, str(level), ha="center", va="bottom",
                fontsize=8, color="#999999")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 5.5)
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              fontsize=9, framealpha=0.9)

    fig.tight_layout()
    _save(fig, "12_radar_overview.png")


def generate_all_visualisations(results: list, l1_data: dict):
    IMAGES_DIR.mkdir(exist_ok=True)
    print(f"\nGenerating visualisations → {IMAGES_DIR}/")
    fig_score_distributions(results)
    fig_scores_by_persona(results)
    fig_layer1_vs_layer2(results, l1_data)
    fig_bottom_conversations(results)
    fig_radar_overview(results)
    print("  ✓  All 5 Layer 2 figures saved.\n")

# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Layer 2 LLM-as-judge evaluation of synthetic banking conversations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ANTHROPIC_API_KEY"),
        help="Anthropic API key. Defaults to ANTHROPIC_API_KEY env var.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model to use for judging. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N conversations (useful for testing prompt quality).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cached results and re-evaluate all conversations.",
    )
    parser.add_argument(
        "--graphs-only",
        action="store_true",
        help="Skip API calls; regenerate graphs from existing CSV only.",
    )
    args = parser.parse_args()

    # ── Load conversations ──────────────────────────────────────────────────
    dump_files = sorted(DUMPS_DIR.glob("conversation_*.txt"))
    if args.limit:
        dump_files = dump_files[: args.limit]

    print(f"Parsing {len(dump_files)} conversation(s) …")
    conversations = []
    for f in dump_files:
        conv = parse_conversation(f)
        if conv:
            conversations.append(conv)
    print(f"  → {len(conversations)} parsed.")

    # ── Graphs-only mode: load existing CSV ────────────────────────────────
    if args.graphs_only:
        if not OUTPUT_CSV.exists():
            print(f"Error: {OUTPUT_CSV} not found. Run without --graphs-only first.")
            return
        results = []
        with open(OUTPUT_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                r = Layer2Result(conv_idx=int(row["conv_idx"]))
                for field in ["knowledge_level","emotional_state","communication_style",
                               "goal_clarity","certainty","model_used","parse_error",
                               "standout_issue","standout_quality",
                               "depth_first_rationale","uncertainty_rationale",
                               "info_drip_rationale","pragmatic_rationale","persona_rationale"]:
                    setattr(r, field, row.get(field, ""))
                for field in ["depth_first_score","uncertainty_score","info_drip_score",
                               "pragmatic_score","persona_score","overall_score"]:
                    try:
                        setattr(r, field, float(row.get(field, 0)))
                    except ValueError:
                        pass
                results.append(r)
        print(f"Loaded {len(results)} results from {OUTPUT_CSV}")
        generate_all_visualisations(results, load_layer1())
        return

    # ── API mode ───────────────────────────────────────────────────────────
    if not args.api_key:
        parser.error("--api-key required (or set ANTHROPIC_API_KEY env var).")

    client = anthropic.Anthropic(api_key=args.api_key)
    l1_data = load_layer1()

    results = []
    cached_count = 0
    api_count    = 0

    for i, conv in enumerate(conversations, 1):
        cached = None if args.force else load_cache(conv.idx)

        if cached:
            cached_count += 1
            judge = cached
            source = "cache"
        else:
            print(f"  [{i:3d}/{len(conversations)}] Conv {conv.idx:03d}  "
                  f"({conv.emotional_state}, {conv.knowledge_level}) …", end=" ", flush=True)
            judge = call_judge(client, conv, args.model)
            save_cache(conv.idx, judge)
            api_count += 1
            source = "api"

            if "_parse_error" in judge:
                print(f"⚠  PARSE ERROR: {judge['_parse_error'][:60]}")
            else:
                r_preview = compute_result(conv, judge, args.model)
                print(f"✓  overall={r_preview.overall_score:.2f}  "
                      f"[d={r_preview.depth_first_score:.0f} "
                      f"u={r_preview.uncertainty_score:.0f} "
                      f"i={r_preview.info_drip_score:.0f} "
                      f"p={r_preview.pragmatic_score:.0f} "
                      f"pe={r_preview.persona_score:.0f}]")

            time.sleep(CALL_DELAY_SEC)

        result = compute_result(conv, judge, args.model)
        results.append(result)

    print(f"\n  API calls: {api_count}   Cached: {cached_count}")

    print_summary(results)
    write_csv(results, OUTPUT_CSV)
    generate_all_visualisations(results, l1_data)


if __name__ == "__main__":
    main()
