#!/usr/bin/env python3
"""
generate_report.py
Reads the CSVs and images produced by eval_predictions.py and writes
a self-contained Markdown report: evaluation_report.md
"""

import csv
import math
from pathlib import Path

BASE_DIR    = Path(__file__).parent
IMAGES_DIR  = BASE_DIR / "images"
REPORT_PATH = BASE_DIR / "evaluation_report.md"

DETAILED_CSV   = BASE_DIR / "prediction_metrics_detailed.csv"
CATEGORY_CSV   = BASE_DIR / "category_summary.csv"
COMPARISON_CSV = BASE_DIR / "model_comparison.csv"

LORA_VARIANTS = ["lora_4b", "lora_12b", "lora_27b"]
ALL_VARIANTS  = ["base_4b", "lora_4b", "base_12b", "lora_12b", "base_27b", "lora_27b"]

# ─────────────────────────────────────────────────────────────────────────────
#  CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_overall(category_rows: list[dict]) -> dict[str, dict]:
    """Return {variant: row} for slice_dim=overall rows."""
    return {
        r["model_variant"]: r
        for r in category_rows
        if r["slice_dim"] == "overall"
    }


def get_slice(category_rows: list[dict], dim: str) -> dict[str, dict[str, dict]]:
    """Return {variant: {slice_val: row}} for a given slice_dim."""
    out: dict[str, dict[str, dict]] = {}
    for r in category_rows:
        if r["slice_dim"] != dim:
            continue
        out.setdefault(r["model_variant"], {})[r["slice_val"]] = r
    return out


def f(val, fmt=".3f", fallback="—"):
    try:
        return format(float(val), fmt) if val not in (None, "", "None") else fallback
    except (TypeError, ValueError):
        return fallback


def pct(val, fallback="—"):
    try:
        return f"{float(val) * 100:.1f}%" if val not in (None, "", "None") else fallback
    except (TypeError, ValueError):
        return fallback


def sig_stars(p_val):
    try:
        p = float(p_val)
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "ns"
    except (TypeError, ValueError):
        return "—"


def img(name: str) -> str:
    path = IMAGES_DIR / name
    if path.exists():
        return f"![{name}](images/{name})"
    return f"_Image not found: `images/{name}`_"


# ─────────────────────────────────────────────────────────────────────────────
#  Section builders
# ─────────────────────────────────────────────────────────────────────────────

def section_overview(overall: dict[str, dict]) -> str:
    lines = [
        "## 1. Overall Performance\n",
        "Core text-generation metrics averaged across all examples. "
        "BLEU, ROUGE-L F1, and METEOR are computed on non-EOS examples only "
        "(EOS examples have no text to compare).\n",
        "| Model | N | BLEU | ROUGE-L F1 | METEOR | Exact Match | EOS Recall | Style Compliance |",
        "|-------|---|------|------------|--------|-------------|------------|-----------------|",
    ]
    for v in ALL_VARIANTS:
        r = overall.get(v)
        if not r:
            continue
        lines.append(
            f"| {v} | {r['n']} "
            f"| {f(r['mean_bleu'])} "
            f"| {f(r['mean_rouge_l_f1'])} "
            f"| {f(r['mean_meteor'])} "
            f"| {pct(r['norm_exact_match_rate'])} "
            f"| {pct(r['eos_recall'])} "
            f"| {f(r['mean_style_compliance'])} |"
        )

    # Narrative: best LoRA variant by BLEU
    best = max(
        ((v, overall[v]) for v in LORA_VARIANTS if v in overall),
        key=lambda x: float(x[1]["mean_bleu"] or 0),
        default=(None, None),
    )
    if best[0]:
        bv, br = best
        lines.append(
            f"\n> **Best LoRA variant by BLEU:** `{bv}` "
            f"(BLEU {f(br['mean_bleu'])}, ROUGE-L {f(br['mean_rouge_l_f1'])}, "
            f"METEOR {f(br['mean_meteor'])})"
        )

    # EOS recall note
    eos_vals = {v: float(overall[v]["eos_recall"] or 0) for v in ALL_VARIANTS if v in overall}
    best_eos = max(eos_vals, key=eos_vals.get)
    worst_eos = min(eos_vals, key=eos_vals.get)
    if eos_vals[best_eos] != eos_vals[worst_eos]:
        lines.append(
            f"\n> **EOS recall** ranges from `{pct(eos_vals[worst_eos])}` ({worst_eos}) "
            f"to `{pct(eos_vals[best_eos])}` ({best_eos})."
        )

    return "\n".join(lines)


def section_eos(category_rows: list[dict], detailed_rows: list[dict]) -> str:
    turn_slice = get_slice(category_rows, "turn_type")

    lines = [
        "## 2. EOS Prediction Analysis\n",
        "EOS examples are turns where the expected model output is `<eos>` — "
        "meaning the conversation should end. The correct prediction is an empty string `\"\"`.\n",
        "| Model | EOS examples | EOS Recall | False Positive Rate |",
        "|-------|-------------|------------|---------------------|",
    ]

    for v in ALL_VARIANTS:
        eos_row  = (turn_slice.get(v) or {}).get("eos_examples", {})
        text_row = (turn_slice.get(v) or {}).get("text_examples", {})
        n_eos    = eos_row.get("n", "—")
        recall   = pct(eos_row.get("eos_recall"))
        fp_rate  = pct(text_row.get("eos_false_pos_rate"))
        lines.append(f"| {v} | {n_eos} | {recall} | {fp_rate} |")

    # Wrong predictions sample
    wrong = [
        r for r in detailed_rows
        if r.get("is_eos_example") == "True" and r.get("eos_correct") == "False"
    ]
    if wrong:
        lines.append("\n### Missed EOS predictions (sample)\n")
        lines.append("| Model | Example idx | Predicted output |")
        lines.append("|-------|-------------|-----------------|")
        for r in wrong[:10]:
            pred = r["predicted_output"][:80].replace("|", "\\|")
            lines.append(f"| {r['model_variant']} | {r['example_idx']} | `{pred}` |")
    else:
        lines.append("\n> All models correctly predicted EOS for every EOS example.")

    lines.append(f"\n{img('15_eos_analysis.png')}\n")
    lines.append(
        "_Panel A: EOS recall per model (fraction of EOS examples correctly predicted as empty). "
        "Panel B: false positive rate (fraction of text examples wrongly predicted as empty). "
        "Panel C: word-count distribution of predictions on EOS-expected examples — "
        "0 words is correct._"
    )
    return "\n".join(lines)


def section_persona(category_rows: list[dict]) -> str:
    dims = [
        ("communication_style", "Communication Style"),
        ("emotional_state",     "Emotional State"),
        ("knowledge_level",     "Knowledge Level"),
        ("goal_clarity",        "Goal Clarity"),
    ]

    lines = [
        "## 3. Performance by Persona Category\n",
        "BLEU scores broken down by persona attribute (LoRA variants, non-EOS examples only).\n",
    ]

    for dim, label in dims:
        slice_data = get_slice(category_rows, dim)
        all_vals = sorted({
            val
            for v in LORA_VARIANTS
            for val in (slice_data.get(v) or {}).keys()
        })
        if not all_vals:
            continue

        lines.append(f"### {label}\n")
        header = "| " + label + " | " + " | ".join(LORA_VARIANTS) + " |"
        sep    = "|---|" + "---|" * len(LORA_VARIANTS)
        lines.append(header)
        lines.append(sep)

        for val in all_vals:
            row_parts = [val]
            for v in LORA_VARIANTS:
                bleu = (slice_data.get(v) or {}).get(val, {}).get("mean_bleu")
                row_parts.append(f(bleu))
            lines.append("| " + " | ".join(row_parts) + " |")

        # Hardest / easiest slice
        bleu_by_val = {}
        for val in all_vals:
            vals_across_models = [
                float((slice_data.get(v) or {}).get(val, {}).get("mean_bleu") or 0)
                for v in LORA_VARIANTS
                if (slice_data.get(v) or {}).get(val, {}).get("mean_bleu") not in (None, "", "None")
            ]
            if vals_across_models:
                bleu_by_val[val] = sum(vals_across_models) / len(vals_across_models)

        if bleu_by_val:
            easiest = max(bleu_by_val, key=bleu_by_val.get)
            hardest = min(bleu_by_val, key=bleu_by_val.get)
            if easiest != hardest:
                lines.append(
                    f"\n> Easiest: **{easiest}** (avg BLEU {bleu_by_val[easiest]:.3f})  "
                    f"· Hardest: **{hardest}** (avg BLEU {bleu_by_val[hardest]:.3f})\n"
                )

    lines.append(f"\n{img('13_category_heatmap.png')}\n")
    lines.append(
        "_Heatmap of mean BLEU per persona attribute value × LoRA model variant. "
        "Green = high BLEU, red = low. Reveals which persona types each model size "
        "handles well or struggles with._"
    )
    return "\n".join(lines)


def section_model_comparison(comparison_rows: list[dict]) -> str:
    lines = [
        "## 4. Cross-Model Comparison\n",
        "Pairwise win rates and Wilcoxon signed-rank test significance "
        "(paired, non-parametric). Win rate = fraction of examples where model A "
        "has higher BLEU than model B.\n",
        "| Model A | Model B | A win rate | B win rate | Tie rate | "
        "Δ BLEU (A−B) | p-value | Significant |",
        "|---------|---------|-----------|-----------|----------|"
        "------------|---------|------------|",
    ]

    lora_rows = [
        r for r in comparison_rows
        if r["model_a"] in LORA_VARIANTS and r["model_b"] in LORA_VARIANTS
    ]
    other_rows = [r for r in comparison_rows if r not in lora_rows]

    for r in lora_rows + other_rows:
        delta = float(r["mean_bleu_a"] or 0) - float(r["mean_bleu_b"] or 0)
        stars = sig_stars(r["p_value"])
        lines.append(
            f"| {r['model_a']} | {r['model_b']} "
            f"| {pct(r['a_win_rate'])} | {pct(r['b_win_rate'])} | {pct(r['tie_rate'])} "
            f"| {delta:+.4f} | {f(r['p_value'], '.4f')} | {stars} |"
        )

    lines.append(f"\n{img('14_model_comparison.png')}\n")
    lines.append(
        "_Left: pairwise win-rate matrix for LoRA variants (diagonal = mean BLEU). "
        "Right: mean BLEU difference with Wilcoxon significance markers "
        "(`*` p<0.05, `**` p<0.01, `***` p<0.001, `ns` = not significant)._"
    )
    return "\n".join(lines)


def section_efficiency(overall: dict[str, dict]) -> str:
    lines = [
        "## 5. Efficiency Frontier\n",
        "Mean BLEU / ROUGE-L vs model scale (parameters). "
        "Bubble size reflects mean generation time.\n",
        "| Model | Params (B) | Mean BLEU | Mean ROUGE-L | Mean gen time (s) |",
        "|-------|-----------|-----------|-------------|-------------------|",
    ]

    param_map = {"4b": 4, "12b": 12, "27b": 27}
    for v in ALL_VARIANTS:
        r = overall.get(v)
        if not r:
            continue
        size_tag = v.split("_", 1)[1]
        params = param_map.get(size_tag, "—")
        lines.append(
            f"| {v} | {params} "
            f"| {f(r['mean_bleu'])} "
            f"| {f(r['mean_rouge_l_f1'])} "
            f"| {f(r['mean_gen_time_sec'])} |"
        )

    # Does LoRA consistently beat base?
    for size_tag in ["4b", "12b", "27b"]:
        base = overall.get(f"base_{size_tag}")
        lora = overall.get(f"lora_{size_tag}")
        if base and lora:
            diff = float(lora["mean_bleu"] or 0) - float(base["mean_bleu"] or 0)
            direction = "outperforms" if diff > 0 else "underperforms vs"
            lines.append(
                f"\n> `lora_{size_tag}` {direction} `base_{size_tag}` "
                f"by {abs(diff):.4f} BLEU points."
            )

    lines.append(f"\n{img('16_efficiency_frontier.png')}\n")
    lines.append(
        "_Scatter plots of mean BLEU (left) and ROUGE-L F1 (right) against model size. "
        "Triangles = LoRA fine-tuned, circles = base. "
        "Dashed/dotted lines are trend lines for each variant type._"
    )
    return "\n".join(lines)


def section_style(category_rows: list[dict]) -> str:
    slice_data = get_slice(category_rows, "communication_style")
    styles = ["terse", "direct", "indirect"]

    lines = [
        "## 6. Style Compliance\n",
        "Rule-based check of whether predictions match the persona's communication style. "
        "Terse = ≤12 words, Direct = ≤20 words, Indirect = ≥8 words with hedging language. "
        "Score 0–1 (higher = more compliant).\n",
        "| Style | " + " | ".join(LORA_VARIANTS) + " |",
        "|-------|" + "---|" * len(LORA_VARIANTS),
    ]

    for style in styles:
        row_parts = [style]
        for v in LORA_VARIANTS:
            score = (slice_data.get(v) or {}).get(style, {}).get("mean_style_compliance")
            row_parts.append(f(score))
        lines.append("| " + " | ".join(row_parts) + " |")

    # Hardest style to comply with
    avg_by_style = {}
    for style in styles:
        vals = [
            float((slice_data.get(v) or {}).get(style, {}).get("mean_style_compliance") or 0)
            for v in LORA_VARIANTS
            if (slice_data.get(v) or {}).get(style, {}).get("mean_style_compliance") not in (None, "", "None")
        ]
        if vals:
            avg_by_style[style] = sum(vals) / len(vals)

    if avg_by_style:
        hardest = min(avg_by_style, key=avg_by_style.get)
        easiest = max(avg_by_style, key=avg_by_style.get)
        lines.append(
            f"\n> Hardest style to comply with: **{hardest}** "
            f"(avg score {avg_by_style[hardest]:.3f})  "
            f"· Easiest: **{easiest}** (avg score {avg_by_style[easiest]:.3f})"
        )

    lines.append(f"\n{img('17_style_compliance.png')}\n")
    lines.append(
        "_Left: grouped bar chart of style compliance score per communication style × model. "
        "Right: box plots of length ratio error per style — 0 means the prediction matched "
        "the expected token count exactly._"
    )
    return "\n".join(lines)


def section_loss(overall: dict[str, dict]) -> str:
    lines = [
        "## 7. Loss & Perplexity\n",
        "Mean cross-entropy loss and perplexity across all examples (including EOS).\n",
        "| Model | Mean Loss | Mean Perplexity |",
        "|-------|-----------|----------------|",
    ]
    for v in ALL_VARIANTS:
        r = overall.get(v)
        if not r:
            continue
        lines.append(f"| {v} | {f(r['mean_loss'])} | {f(r['mean_perplexity'])} |")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    for path in [DETAILED_CSV, CATEGORY_CSV, COMPARISON_CSV]:
        if not path.exists():
            print(f"ERROR: {path.name} not found. Run eval_predictions.py first.")
            return

    print("Loading CSVs …")
    detailed_rows   = load_csv(DETAILED_CSV)
    category_rows   = load_csv(CATEGORY_CSV)
    comparison_rows = load_csv(COMPARISON_CSV)

    overall = get_overall(category_rows)

    n_examples = int(overall.get("lora_4b", {}).get("n") or 0)
    n_eos      = int(overall.get("lora_4b", {}).get("n_eos") or 0)
    n_models   = len([v for v in ALL_VARIANTS if v in overall])

    print("Building report …")

    sections = [
        f"# Prediction Evaluation Report\n",
        f"**Dataset:** {n_examples} validation examples "
        f"({n_eos} EOS · {n_examples - n_eos} text)  \n"
        f"**Model variants evaluated:** {n_models} "
        f"({len(LORA_VARIANTS)} LoRA fine-tuned + {n_models - len(LORA_VARIANTS)} base)  \n"
        f"**Metrics:** BLEU · ROUGE-L · METEOR · Exact Match · EOS Recall · "
        f"Style Compliance · Loss · Perplexity\n",
        "---\n",
        section_overview(overall),
        "\n---\n",
        section_eos(category_rows, detailed_rows),
        "\n---\n",
        section_persona(category_rows),
        "\n---\n",
        section_model_comparison(comparison_rows),
        "\n---\n",
        section_efficiency(overall),
        "\n---\n",
        section_style(category_rows),
        "\n---\n",
        section_loss(overall),
    ]

    report = "\n".join(sections)

    with open(REPORT_PATH, "w", encoding="utf-8") as f_out:
        f_out.write(report)

    print(f"  ✓  Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
