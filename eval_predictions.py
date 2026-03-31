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
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import numpy as np
import seaborn as sns
from scipy.stats import wilcoxon

import nltk
for _res in ("wordnet", "omw-1.4"):
    nltk.download(_res, quiet=True)
from nltk.translate.meteor_score import meteor_score as _nltk_meteor

# ─────────────────────────────────────────────────────────────────────────────
#  HARDCODED PATHS — update these three lines before running
# ─────────────────────────────────────────────────────────────────────────────

MODEL_FILES = {
    "gemma_4b":  Path("predictions_gemma_4b.jsonl"),   # ← update path
    "gemma_12b": Path("predictions_gemma_12b.jsonl"),  # ← update path
    "gemma_27b": Path("predictions_gemma_27b.jsonl"),  # ← update path
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

BASE_DIR          = Path(__file__).parent
IMAGES_DIR        = BASE_DIR / "images"
DETAILED_CSV      = BASE_DIR / "prediction_metrics_detailed.csv"
CATEGORY_CSV      = BASE_DIR / "category_summary.csv"
COMPARISON_CSV    = BASE_DIR / "model_comparison.csv"

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


def style_compliance(predicted_norm: str, persona: dict) -> float:
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

def compute_meteor(reference: str, hypothesis: str) -> float:
    if not reference or not hypothesis:
        return 0.0
    try:
        return round(float(_nltk_meteor([reference.split()], hypothesis.split())), 4)
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_file(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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

    # New metrics
    normalized_exact_match: bool  = False
    eos_correct:            object = None   # bool for EOS examples, None otherwise
    eos_false_pos:          object = None   # bool for non-EOS examples, None otherwise
    length_ratio_error:     float  = 0.0
    style_compliance_score: float  = 0.0
    meteor_score:           float  = 0.0

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
        # New metrics
        normalized_exact_match  = pred_is_eos if is_eos else (norm_pred == norm_exp),
        eos_correct             = pred_is_eos if is_eos else None,
        eos_false_pos           = pred_is_eos if not is_eos else None,
        length_ratio_error      = abs(1.0 - token_ratio) if token_ratio > 0 else 1.0,
        style_compliance_score  = style_compliance(norm_pred, persona),
        meteor_score            = meteor,
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
                  "mean_loss", "mean_perplexity", "norm_exact_match_rate",
                  "eos_recall", "eos_false_pos_rate",
                  "mean_length_ratio_error", "mean_token_ratio",
                  "mean_style_compliance", "mean_gen_time_sec"]
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

def print_summary(all_metrics: dict):
    sep = "─" * 72
    print(f"\n{'=' * 72}")
    print("  PREDICTION EVALUATION SUMMARY")
    print(f"{'=' * 72}")

    print(f"\n{sep}")
    print(f"  {'Model':<18} {'N':>4}  {'BLEU':>6}  {'ROUGE-F1':>8}  "
          f"{'METEOR':>6}  {'NormEM%':>7}  {'EOS-Rec':>8}  {'StyleC':>6}")
    print(f"  {'─'*18}  {'─'*4}  {'─'*6}  {'─'*8}  {'─'*6}  {'─'*7}  {'─'*8}  {'─'*6}")

    for variant in sorted(all_metrics.keys()):
        metrics = all_metrics[variant]
        agg = aggregate(metrics)
        print(f"  {variant:<18} {agg['n']:>4}  "
              f"{agg['mean_bleu']:>6.3f}  "
              f"{agg['mean_rouge_l_f1']:>8.3f}  "
              f"{agg['mean_meteor']:>6.3f}  "
              f"{(agg['norm_exact_match_rate'] or 0)*100:>6.1f}%  "
              f"{(agg['eos_recall'] or 0):>7.1%}  "
              f"{agg['mean_style_compliance']:>6.3f}")

    print(f"\n{sep}")
    print("  EOS ANALYSIS  (16 expected-EOS examples in val set)")
    print(sep)
    for variant in sorted(all_metrics.keys()):
        metrics = all_metrics[variant]
        eos_ex  = [m for m in metrics if m.is_eos_example]
        correct = sum(m.eos_correct for m in eos_ex)
        wrong   = [m for m in eos_ex if not m.eos_correct]
        print(f"  {variant:<20}: {correct}/{len(eos_ex)} correct EOS predictions")
        for m in wrong[:3]:
            print(f"    ✗ Conv {m.example_idx}: predicted → \"{m.predicted_output[:60]}\"")

    print(f"{'=' * 72}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Check files exist ─────────────────────────────────────────────────
    missing = [name for name, path in MODEL_FILES.items() if not path.exists()]
    if missing:
        print(f"ERROR: Missing prediction files for: {missing}")
        print("Update the MODEL_FILES paths at the top of this script and re-run.")
        return

    # ── Load + verify alignment ───────────────────────────────────────────
    print("Loading prediction files …")
    files_data = {name: load_file(path) for name, path in MODEL_FILES.items()}

    try:
        n_examples = verify_alignment(files_data)
    except ValueError as e:
        print(f"ERROR: {e}")
        return

    # ── Compute metrics ───────────────────────────────────────────────────
    print(f"\nComputing metrics for {n_examples} examples × {len(MODEL_FILES)} files …")
    all_flat: list = []          # flat list of ExampleMetrics (all models)
    all_metrics: dict = {}       # {variant_name: [ExampleMetrics]}

    for model_name, rows in files_data.items():
        for model_key, variant_prefix in [("base_model", "base"), ("lora_model", "lora")]:
            size_tag   = model_name.split("_")[1]        # "4b" / "12b" / "27b"
            variant_name = f"{variant_prefix}_{size_tag}"
            variant_metrics = []
            for idx, row in enumerate(rows):
                if model_key not in row:
                    continue
                m = compute_metrics_for_example(row, model_key, model_name, idx)
                variant_metrics.append(m)
                all_flat.append(m)
            all_metrics[variant_name] = variant_metrics
            print(f"  {variant_name:<18}: {len(variant_metrics)} examples")

    print(f"\n  Total metric rows: {len(all_flat)}")

    # ── Cross-model comparison ─────────────────────────────────────────────
    print("\nRunning cross-model comparison (Wilcoxon tests) …")
    comparison_rows = compare_models(all_metrics)

    # ── Print summary ─────────────────────────────────────────────────────
    print_summary(all_metrics)

    # ── Write CSVs ────────────────────────────────────────────────────────
    print("Writing CSVs …")
    write_detailed_csv(all_flat, DETAILED_CSV)
    write_category_csv(all_metrics, CATEGORY_CSV)
    write_comparison_csv(comparison_rows, COMPARISON_CSV)

    # ── Generate figures ──────────────────────────────────────────────────
    print("\nGenerating visualisations …")
    IMAGES_DIR.mkdir(exist_ok=True)
    fig_category_heatmap(all_metrics)
    fig_model_comparison(all_metrics, comparison_rows)
    fig_eos_analysis(all_metrics)
    fig_efficiency_frontier(all_metrics)
    fig_style_compliance(all_metrics)
    print("  ✓  All 5 figures saved.\n")


if __name__ == "__main__":
    main()
