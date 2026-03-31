#!/usr/bin/env python3
"""
generate_report.py
Reads the CSVs and images produced by eval_predictions.py and writes
a self-contained Markdown report: evaluation_report.md

Designed to be **flexible**: detects which metrics are actually populated
in the CSVs and only reports on those. Sections for BERTScore, BLEURT,
LLM-as-judge, entity analysis, behavioural signals, and conversation-level
metrics are included only when the underlying data exists.

Primary metric hierarchy (first available is used as "headline" metric):
  BERTScore F1 > BLEURT > METEOR > BLEU
"""

import csv
from pathlib import Path

BASE_DIR    = Path(__file__).parent
IMAGES_DIR  = BASE_DIR / "images"
REPORT_PATH = BASE_DIR / "evaluation_report.md"

DETAILED_CSV      = BASE_DIR / "prediction_metrics_detailed.csv"
CATEGORY_CSV      = BASE_DIR / "category_summary.csv"
COMPARISON_CSV    = BASE_DIR / "model_comparison.csv"
CONVERSATION_CSV  = BASE_DIR / "conversation_metrics.csv"

LORA_VARIANTS = ["lora_4b", "lora_12b", "lora_27b"]
ALL_VARIANTS  = ["base_4b", "lora_4b", "base_12b", "lora_12b", "base_27b", "lora_27b"]

# ── Metric hierarchy: first available is the "primary" metric ────────────
# Each entry: (csv_col_in_category, csv_col_in_detailed, display_name)
METRIC_HIERARCHY = [
    ("mean_bertscore_f1",  "bertscore_f1",  "BERTScore F1"),
    ("mean_bleurt",        "bleurt_score",   "BLEURT"),
    ("mean_meteor",        "meteor_score",   "METEOR"),
    ("mean_bleu",          "bleu_score",     "BLEU"),
]


# ─────────────────────────────────────────────────────────────────────────────
#  CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


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


def col_populated(rows: list[dict], col: str) -> bool:
    """True if *any* row has a non-empty, non-None value for col."""
    for r in rows:
        v = r.get(col, "")
        if v not in (None, "", "None", "nan"):
            try:
                float(v)
                return True
            except (ValueError, TypeError):
                continue
    return False


def f(val, fmt=".3f", fallback="—"):
    try:
        return format(float(val), fmt) if val not in (None, "", "None", "nan") else fallback
    except (TypeError, ValueError):
        return fallback


def pct(val, fallback="—"):
    try:
        return f"{float(val) * 100:.1f}%" if val not in (None, "", "None", "nan") else fallback
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
    return ""  # silently omit missing images


def detect_primary_metric(overall: dict[str, dict]) -> tuple[str, str, str]:
    """Return (category_col, detailed_col, display_name) for the best available metric."""
    sample_row = next(iter(overall.values()), {})
    for cat_col, det_col, name in METRIC_HIERARCHY:
        val = sample_row.get(cat_col, "")
        if val not in (None, "", "None", "nan"):
            try:
                float(val)
                return cat_col, det_col, name
            except (ValueError, TypeError):
                continue
    return "mean_bleu", "bleu_score", "BLEU"


def variants_present(overall: dict[str, dict]) -> list[str]:
    """Return only variants that exist in the data, in canonical order."""
    return [v for v in ALL_VARIANTS if v in overall]


def lora_variants_present(overall: dict[str, dict]) -> list[str]:
    return [v for v in LORA_VARIANTS if v in overall]


# ─────────────────────────────────────────────────────────────────────────────
#  Section builders
# ─────────────────────────────────────────────────────────────────────────────

def section_overview(overall: dict[str, dict], category_rows: list[dict]) -> str:
    """Section 1: Overall performance table with all available surface + semantic metrics."""
    primary_cat, _, primary_name = detect_primary_metric(overall)
    variants = variants_present(overall)

    # Build column list dynamically
    metric_cols = []  # (category_col, display_name, format_fn)
    for cat_col, _, name in METRIC_HIERARCHY:
        if col_populated([overall[v] for v in variants], cat_col):
            metric_cols.append((cat_col, name, lambda v, c=cat_col: f(v.get(c))))

    # Always include if available
    static_cols = [
        ("mean_meteor",           "METEOR",           lambda v: f(v.get("mean_meteor"))),
        ("norm_exact_match_rate", "Exact Match",      lambda v: pct(v.get("norm_exact_match_rate"))),
        ("eos_recall",            "EOS Recall",       lambda v: pct(v.get("eos_recall"))),
        ("mean_style_compliance", "Style Compliance", lambda v: f(v.get("mean_style_compliance"))),
    ]

    # Deduplicate: remove static cols that are already in metric_cols
    metric_col_names = {mc[0] for mc in metric_cols}
    for sc in static_cols:
        if sc[0] not in metric_col_names:
            metric_cols.append(sc)

    header_names = ["Model", "N"] + [mc[1] for mc in metric_cols]
    lines = [
        "## 1. Overall Performance\n",
        f"Primary semantic metric: **{primary_name}** "
        "(first available in hierarchy: BERTScore F1 → BLEURT → METEOR → BLEU). "
        "Surface lexical metrics (BLEU, ROUGE) are shown for reference but should not "
        "be interpreted as primary indicators of generation quality.\n",
        "| " + " | ".join(header_names) + " |",
        "|" + "|".join(["---"] * len(header_names)) + "|",
    ]

    for v in variants:
        r = overall[v]
        cells = [v, r.get("n", "—")]
        for _, _, fmt_fn in metric_cols:
            cells.append(fmt_fn(r))
        lines.append("| " + " | ".join(cells) + " |")

    # Best LoRA variant by primary metric
    lora_vars = lora_variants_present(overall)
    if lora_vars:
        best_v, best_r = max(
            ((v, overall[v]) for v in lora_vars),
            key=lambda x: float(x[1].get(primary_cat) or 0),
        )
        lines.append(
            f"\n> **Best LoRA variant by {primary_name}:** `{best_v}` "
            f"({primary_name} = {f(best_r.get(primary_cat))})"
        )

    return "\n".join(lines)


def section_semantic(overall: dict[str, dict], detailed_rows: list[dict]) -> str | None:
    """Section 2: Semantic similarity metrics (BERTScore, BLEURT). Omitted if neither available."""
    has_bert = col_populated(detailed_rows, "bertscore_f1")
    has_bleurt = col_populated(detailed_rows, "bleurt_score")
    if not has_bert and not has_bleurt:
        return None

    variants = variants_present(overall)
    lines = [
        "## 2. Semantic Similarity\n",
        "These metrics use neural models to measure meaning-level similarity between "
        "predicted and reference responses, going beyond surface lexical overlap.\n",
    ]

    # BERTScore table
    if has_bert:
        lines.append("### BERTScore\n")
        lines.append(
            "Computes token-level contextual embedding similarity using DeBERTa-xlarge-MNLI. "
            "Values are rescaled to [0, 1] with baseline correction.\n"
        )
        lines.append("| Model | F1 | Precision | Recall |")
        lines.append("|-------|-----|-----------|--------|")
        for v in variants:
            r = overall[v]
            lines.append(
                f"| {v} | {f(r.get('mean_bertscore_f1'))} "
                f"| {f(r.get('mean_bertscore_precision'))} "
                f"| {f(r.get('mean_bertscore_recall'))} |"
            )

        bert_img = img("18_bertscore_comparison.png")
        if bert_img:
            lines.append(f"\n{bert_img}\n")
            lines.append(
                "_BERTScore F1/Precision/Recall across model variants. "
                "Higher F1 indicates better semantic alignment with reference responses._"
            )

    # BLEURT table
    if has_bleurt:
        lines.append("\n### BLEURT\n")
        lines.append(
            "Learned evaluation metric trained on human quality judgments. "
            "Score ≈ 0 indicates reference-quality text; negative = worse, positive = better.\n"
        )
        lines.append("| Model | BLEURT |")
        lines.append("|-------|--------|")
        for v in variants:
            lines.append(f"| {v} | {f(overall[v].get('mean_bleurt'))} |")

    return "\n".join(lines)


def section_eos(category_rows: list[dict], detailed_rows: list[dict]) -> str:
    """Section 3: EOS prediction analysis."""
    turn_slice = get_slice(category_rows, "turn_type")
    variants = [v for v in ALL_VARIANTS if v in {r["model_variant"] for r in category_rows}]

    lines = [
        "## 3. EOS Prediction Analysis\n",
        "EOS examples are turns where the expected output is `<eos>` — "
        "the conversation should end. Correct prediction = empty string.\n",
        "| Model | EOS examples | EOS Recall | False Positive Rate |",
        "|-------|-------------|------------|---------------------|",
    ]

    for v in variants:
        eos_row  = (turn_slice.get(v) or {}).get("eos_examples", {})
        text_row = (turn_slice.get(v) or {}).get("text_examples", {})
        n_eos    = eos_row.get("n", "—")
        recall   = pct(eos_row.get("eos_recall"))
        fp_rate  = pct(text_row.get("eos_false_pos_rate"))
        lines.append(f"| {v} | {n_eos} | {recall} | {fp_rate} |")

    # Missed EOS sample
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

    eos_img = img("15_eos_analysis.png")
    if eos_img:
        lines.append(f"\n{eos_img}\n")
        lines.append(
            "_Panel A: EOS recall per model. Panel B: false positive rate. "
            "Panel C: word-count distribution on EOS-expected examples (0 words = correct)._"
        )
    return "\n".join(lines)


def section_behavioural(overall: dict[str, dict], detailed_rows: list[dict]) -> str | None:
    """Section 4: Behavioural signals — hedge rate, promise rate, TTR, role confusion."""
    metrics_needed = ["hedge_rate_predicted", "promise_rate_predicted", "ttr_predicted", "role_confused"]
    if not any(col_populated(detailed_rows, m) for m in metrics_needed):
        return None

    variants = variants_present(overall)
    lines = [
        "## 4. Behavioural Signals\n",
        "Behavioural metrics capture pragmatic properties of generated text that lexical "
        "metrics miss. These are inspired by Wang et al. (2025) findings that LLM-generated "
        "user utterances systematically under-hedge and over-promise compared to real humans.\n",
    ]

    # Main table
    cols = []  # (display, category_col, fmt)
    if col_populated(detailed_rows, "hedge_rate_predicted"):
        cols.append(("Hedge Δ",   "mean_hedge_rate_delta",   lambda r: f(r.get("mean_hedge_rate_delta"), "+.3f")))
    if col_populated(detailed_rows, "promise_rate_predicted"):
        cols.append(("Promise Δ", "mean_promise_rate_delta", lambda r: f(r.get("mean_promise_rate_delta"), "+.3f")))
    if col_populated(detailed_rows, "ttr_predicted"):
        cols.append(("TTR Δ",     "mean_ttr_delta",          lambda r: f(r.get("mean_ttr_delta"), "+.3f")))
    if col_populated(detailed_rows, "role_confused"):
        cols.append(("Role Confusion", "role_confusion_rate", lambda r: pct(r.get("role_confusion_rate"))))

    if cols:
        header = "| Model | " + " | ".join(c[0] for c in cols) + " |"
        sep    = "|---" + "|---" * len(cols) + "|"
        lines.append(header)
        lines.append(sep)
        for v in variants:
            r = overall[v]
            cells = [v] + [c[2](r) for c in cols]
            lines.append("| " + " | ".join(cells) + " |")

    # Interpretation guide
    lines.append(
        "\n> **Reading the deltas:** Hedge Δ and Promise Δ are (predicted − reference). "
        "Negative hedge Δ = model under-hedges vs reference. Positive promise Δ = model over-promises. "
        "TTR Δ measures vocabulary diversity shift. "
        "Role confusion = fraction of turns where the model produces assistant-like language.\n"
    )

    behav_img = img("19_behavioural_signals.png")
    if behav_img:
        lines.append(f"{behav_img}\n")
        lines.append(
            "_Behavioural signal distributions across model variants. "
            "Closer to zero delta = better alignment with reference behaviour._"
        )

    return "\n".join(lines)


def section_entity(overall: dict[str, dict], detailed_rows: list[dict]) -> str | None:
    """Section 5: Entity-level precision/recall/F1."""
    if not col_populated(detailed_rows, "entity_f1"):
        return None

    variants = variants_present(overall)
    lines = [
        "## 5. Entity Analysis\n",
        "Regex-based extraction and comparison of banking-domain entities: "
        "amounts ($X.XX), dates, card numbers (last 4 digits), and merchant names. "
        "Measures whether the model preserves factual details from the conversation context.\n",
        "| Model | Entity Precision | Entity Recall | Entity F1 |",
        "|-------|-----------------|---------------|-----------|",
    ]

    for v in variants:
        r = overall[v]
        lines.append(
            f"| {v} | {f(r.get('mean_entity_precision'))} "
            f"| {f(r.get('mean_entity_recall'))} "
            f"| {f(r.get('mean_entity_f1'))} |"
        )

    # Highlight recall vs precision gap
    for v in lora_variants_present(overall):
        r = overall[v]
        try:
            prec = float(r.get("mean_entity_precision") or 0)
            rec  = float(r.get("mean_entity_recall") or 0)
            if abs(prec - rec) > 0.05:
                direction = "hallucinates entities" if prec < rec else "drops entities"
                lines.append(f"\n> `{v}`: precision={f(r.get('mean_entity_precision'))}, "
                             f"recall={f(r.get('mean_entity_recall'))} — model {direction}.")
        except (TypeError, ValueError):
            pass

    ent_img = img("21_entity_analysis.png")
    if ent_img:
        lines.append(f"\n{ent_img}\n")
        lines.append(
            "_Entity-level precision, recall, and F1 across model variants. "
            "Low precision = entity hallucination; low recall = entity omission._"
        )

    return "\n".join(lines)


def section_judge(overall: dict[str, dict], detailed_rows: list[dict]) -> str | None:
    """Section 6: LLM-as-judge rubric scores. Omitted if no judge data."""
    if not col_populated(detailed_rows, "judge_mean"):
        return None

    variants = variants_present(overall)
    dims = [
        ("mean_judge_semantic",  "Semantic Fidelity"),
        ("mean_judge_persona",   "Persona Voice"),
        ("mean_judge_coherence", "Conversational Coherence"),
        ("mean_judge_goal",      "Goal Directedness"),
        ("mean_judge_realism",   "Human Realism"),
        ("mean_judge_info_cal",  "Information Calibration"),
        ("mean_judge_overall",   "Overall Mean"),
    ]

    # Filter to only populated dimensions
    dims = [(col, name) for col, name in dims if col_populated([overall.get(v, {}) for v in variants], col)]
    if not dims:
        return None

    lines = [
        "## 6. LLM-as-Judge Evaluation\n",
        "Each predicted response is scored 1–5 across six dimensions by an LLM judge "
        "(Claude). Higher is better. The rubric evaluates semantic fidelity, persona voice, "
        "conversational coherence, goal directedness, human realism, and information calibration.\n",
        "| Model | " + " | ".join(d[1] for d in dims) + " |",
        "|---" + "|---" * len(dims) + "|",
    ]

    for v in variants:
        r = overall.get(v, {})
        cells = [v] + [f(r.get(col)) for col, _ in dims]
        lines.append("| " + " | ".join(cells) + " |")

    # Best variant by judge overall
    lora_vars = lora_variants_present(overall)
    if lora_vars and col_populated([overall[v] for v in lora_vars], "mean_judge_overall"):
        best_v = max(lora_vars, key=lambda v: float(overall[v].get("mean_judge_overall") or 0))
        lines.append(
            f"\n> **Best LoRA variant by judge:** `{best_v}` "
            f"(mean = {f(overall[best_v].get('mean_judge_overall'))})"
        )

    # Weakest dimension across all LoRA
    if len(dims) > 1:
        dim_avgs = {}
        for col, name in dims:
            if name == "Overall Mean":
                continue
            vals = [float(overall[v].get(col) or 0) for v in lora_vars
                    if overall[v].get(col) not in (None, "", "None", "nan")]
            if vals:
                dim_avgs[name] = sum(vals) / len(vals)
        if dim_avgs:
            weakest = min(dim_avgs, key=dim_avgs.get)
            strongest = max(dim_avgs, key=dim_avgs.get)
            lines.append(
                f"\n> **Strongest dimension:** {strongest} ({dim_avgs[strongest]:.2f}) · "
                f"**Weakest:** {weakest} ({dim_avgs[weakest]:.2f})"
            )

    judge_img = img("20_judge_radar.png")
    if judge_img:
        lines.append(f"\n{judge_img}\n")
        lines.append("_Radar plot of LLM-as-judge dimension scores per model variant._")

    return "\n".join(lines)


def section_conversation(conversation_rows: list[dict] | None, overall: dict[str, dict]) -> str | None:
    """Section 7: Conversation-level metrics from conversation_metrics.csv."""
    if not conversation_rows:
        return None

    variants = [v for v in ALL_VARIANTS if any(r["model_variant"] == v for r in conversation_rows)]
    if not variants:
        return None

    lines = [
        "## 7. Conversation-Level Metrics\n",
        "Metrics computed per conversation (grouped by generation_idx), capturing "
        "cross-turn coherence and consistency — the properties that turn-level "
        "metrics cannot measure.\n",
    ]

    # Aggregate conversation metrics per variant
    conv_agg: dict[str, dict] = {}
    for v in variants:
        v_rows = [r for r in conversation_rows if r["model_variant"] == v]
        if not v_rows:
            continue
        agg = {}

        for col in ["style_compliance_std", "style_compliance_min", "self_bleu",
                     "hedge_rate_correlation", "length_trajectory_correlation",
                     "hedge_rate_delta", "judge_mean"]:
            vals = []
            for r in v_rows:
                val = r.get(col, "")
                if val not in (None, "", "None", "nan"):
                    try:
                        vals.append(float(val))
                    except (ValueError, TypeError):
                        pass
            agg[col] = sum(vals) / len(vals) if vals else None
            agg[f"{col}_n"] = len(vals)

        # Role confusion: sum across conversations
        rc_vals = []
        for r in v_rows:
            val = r.get("n_role_confused_turns", "")
            if val not in (None, "", "None", "nan"):
                try:
                    rc_vals.append(int(float(val)))
                except (ValueError, TypeError):
                    pass
        agg["total_role_confused"] = sum(rc_vals) if rc_vals else None
        agg["n_conversations"] = len(v_rows)
        conv_agg[v] = agg

    # Build table dynamically based on available data
    col_defs = []  # (display_name, format_fn)

    sample_agg = next(iter(conv_agg.values()), {})

    if sample_agg.get("style_compliance_std") is not None:
        col_defs.append(("Style σ",  lambda a: f(a.get("style_compliance_std"))))
        col_defs.append(("Style min", lambda a: f(a.get("style_compliance_min"))))

    if sample_agg.get("self_bleu") is not None:
        col_defs.append(("Self-BLEU", lambda a: f(a.get("self_bleu"))))

    if sample_agg.get("hedge_rate_correlation") is not None:
        col_defs.append(("Hedge ρ",   lambda a: f(a.get("hedge_rate_correlation"))))

    if sample_agg.get("length_trajectory_correlation") is not None:
        col_defs.append(("Length ρ",   lambda a: f(a.get("length_trajectory_correlation"))))

    if sample_agg.get("judge_mean") is not None:
        col_defs.append(("Judge mean", lambda a: f(a.get("judge_mean"))))

    if sample_agg.get("total_role_confused") is not None:
        col_defs.append(("Role confused turns", lambda a: str(a.get("total_role_confused", "—"))))

    if col_defs:
        header = "| Model | Convs | " + " | ".join(c[0] for c in col_defs) + " |"
        sep    = "|---|---" + "|---" * len(col_defs) + "|"
        lines.append(header)
        lines.append(sep)
        for v in variants:
            a = conv_agg.get(v, {})
            cells = [v, str(a.get("n_conversations", "—"))]
            cells += [c[1](a) for c in col_defs]
            lines.append("| " + " | ".join(cells) + " |")

    lines.append(
        "\n> **Reading this table:** "
        "Style σ = standard deviation of style compliance within a conversation (lower = more consistent). "
        "Self-BLEU = pairwise BLEU among predicted turns in the same conversation "
        "(very high values suggest mode collapse / repetitive outputs). "
        "Hedge ρ / Length ρ = Spearman correlation between predicted and reference trajectories "
        "across conversation turns (higher = better tracking of conversational dynamics).\n"
    )

    conv_img = img("22_conversation_level.png")
    if conv_img:
        lines.append(f"{conv_img}\n")
        lines.append(
            "_Conversation-level metrics distributions. "
            "These capture cross-turn coherence that single-turn metrics cannot measure._"
        )

    return "\n".join(lines)


def section_persona(category_rows: list[dict], overall: dict[str, dict]) -> str:
    """Section 8: Performance by persona category, using primary metric."""
    primary_cat, _, primary_name = detect_primary_metric(overall)
    lora_vars = lora_variants_present(overall)

    dims = [
        ("communication_style", "Communication Style"),
        ("emotional_state",     "Emotional State"),
        ("knowledge_level",     "Knowledge Level"),
        ("goal_clarity",        "Goal Clarity"),
    ]

    lines = [
        "## 8. Performance by Persona Category\n",
        f"{primary_name} scores broken down by persona attribute (LoRA variants, non-EOS examples only).\n",
    ]

    for dim, label in dims:
        slice_data = get_slice(category_rows, dim)
        all_vals = sorted({
            val
            for v in lora_vars
            for val in (slice_data.get(v) or {}).keys()
        })
        if not all_vals:
            continue

        lines.append(f"### {label}\n")
        header = "| " + label + " | " + " | ".join(lora_vars) + " |"
        sep    = "|---|" + "---|" * len(lora_vars)
        lines.append(header)
        lines.append(sep)

        for val in all_vals:
            row_parts = [val]
            for v in lora_vars:
                metric_val = (slice_data.get(v) or {}).get(val, {}).get(primary_cat)
                row_parts.append(f(metric_val))
            lines.append("| " + " | ".join(row_parts) + " |")

        # Easiest / hardest slice
        metric_by_val = {}
        for val in all_vals:
            vals_across = [
                float((slice_data.get(v) or {}).get(val, {}).get(primary_cat) or 0)
                for v in lora_vars
                if (slice_data.get(v) or {}).get(val, {}).get(primary_cat)
                   not in (None, "", "None", "nan")
            ]
            if vals_across:
                metric_by_val[val] = sum(vals_across) / len(vals_across)

        if metric_by_val:
            easiest = max(metric_by_val, key=metric_by_val.get)
            hardest = min(metric_by_val, key=metric_by_val.get)
            if easiest != hardest:
                lines.append(
                    f"\n> Easiest: **{easiest}** (avg {primary_name} {metric_by_val[easiest]:.3f})  "
                    f"· Hardest: **{hardest}** (avg {primary_name} {metric_by_val[hardest]:.3f})\n"
                )

    heatmap_img = img("13_category_heatmap.png")
    if heatmap_img:
        lines.append(f"\n{heatmap_img}\n")
        lines.append(
            f"_Heatmap of mean {primary_name} per persona attribute × LoRA model variant._"
        )
    return "\n".join(lines)


def section_model_comparison(comparison_rows: list[dict], overall: dict[str, dict]) -> str:
    """Section 9: Cross-model pairwise comparison."""
    primary_cat, _, primary_name = detect_primary_metric(overall)

    lines = [
        "## 9. Cross-Model Comparison\n",
        "Pairwise win rates and Wilcoxon signed-rank test significance "
        "(paired, non-parametric). Win rate = fraction of examples where model A "
        "outscores model B on BLEU.\n",
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

    lines.append(
        "\n> **Note:** This comparison uses BLEU for pairwise ranking because it was the "
        "metric used in the Wilcoxon test. See sections above for semantic and judge-based comparisons.\n"
    )

    comp_img = img("14_model_comparison.png")
    if comp_img:
        lines.append(f"{comp_img}\n")
        lines.append(
            "_Left: pairwise win-rate matrix (diagonal = mean score). "
            "Right: mean difference with Wilcoxon significance markers._"
        )
    return "\n".join(lines)


def section_efficiency(overall: dict[str, dict]) -> str:
    """Section 10: Efficiency frontier — quality vs scale."""
    primary_cat, _, primary_name = detect_primary_metric(overall)
    variants = variants_present(overall)

    lines = [
        "## 10. Efficiency Frontier\n",
        f"Mean {primary_name} vs model scale (parameters). "
        "Bubble size reflects mean generation time.\n",
        f"| Model | Params (B) | {primary_name} | METEOR | Mean gen time (s) |",
        "|-------|-----------|-----------|--------|-------------------|",
    ]

    param_map = {"4b": 4, "12b": 12, "27b": 27}
    for v in variants:
        r = overall[v]
        size_tag = v.split("_", 1)[1]
        params = param_map.get(size_tag, "—")
        lines.append(
            f"| {v} | {params} "
            f"| {f(r.get(primary_cat))} "
            f"| {f(r.get('mean_meteor'))} "
            f"| {f(r.get('mean_gen_time_sec'))} |"
        )

    # LoRA vs base delta
    for size_tag in ["4b", "12b", "27b"]:
        base = overall.get(f"base_{size_tag}")
        lora = overall.get(f"lora_{size_tag}")
        if base and lora:
            try:
                diff = float(lora.get(primary_cat) or 0) - float(base.get(primary_cat) or 0)
                direction = "outperforms" if diff > 0 else "underperforms vs"
                lines.append(
                    f"\n> `lora_{size_tag}` {direction} `base_{size_tag}` "
                    f"by {abs(diff):.4f} {primary_name} points."
                )
            except (TypeError, ValueError):
                pass

    eff_img = img("16_efficiency_frontier.png")
    if eff_img:
        lines.append(f"\n{eff_img}\n")
        lines.append(
            "_Scatter plots of quality metric against model size. "
            "Triangles = LoRA fine-tuned, circles = base._"
        )
    return "\n".join(lines)


def section_style(category_rows: list[dict], overall: dict[str, dict]) -> str:
    """Section 11: Style compliance analysis."""
    lora_vars = lora_variants_present(overall)
    slice_data = get_slice(category_rows, "communication_style")
    styles = sorted({
        val
        for v in lora_vars
        for val in (slice_data.get(v) or {}).keys()
    })
    if not styles:
        return ""

    lines = [
        "## 11. Style Compliance\n",
        "Rule-based check of whether predictions match the persona's communication style. "
        "Terse = ≤12 words, Direct = ≤20 words, Indirect = ≥8 words with hedging language. "
        "Score 0–1 (higher = more compliant).\n",
        "| Style | " + " | ".join(lora_vars) + " |",
        "|-------|" + "---|" * len(lora_vars),
    ]

    for style in styles:
        row_parts = [style]
        for v in lora_vars:
            score = (slice_data.get(v) or {}).get(style, {}).get("mean_style_compliance")
            row_parts.append(f(score))
        lines.append("| " + " | ".join(row_parts) + " |")

    # Hardest style
    avg_by_style = {}
    for style in styles:
        vals = [
            float((slice_data.get(v) or {}).get(style, {}).get("mean_style_compliance") or 0)
            for v in lora_vars
            if (slice_data.get(v) or {}).get(style, {}).get("mean_style_compliance")
               not in (None, "", "None", "nan")
        ]
        if vals:
            avg_by_style[style] = sum(vals) / len(vals)

    if avg_by_style:
        hardest = min(avg_by_style, key=avg_by_style.get)
        easiest = max(avg_by_style, key=avg_by_style.get)
        lines.append(
            f"\n> Hardest: **{hardest}** (avg {avg_by_style[hardest]:.3f}) · "
            f"Easiest: **{easiest}** (avg {avg_by_style[easiest]:.3f})"
        )

    style_img = img("17_style_compliance.png")
    if style_img:
        lines.append(f"\n{style_img}\n")
        lines.append(
            "_Left: style compliance per communication style × model. "
            "Right: length ratio error distribution per style._"
        )
    return "\n".join(lines)


def section_metric_commentary(overall: dict[str, dict], detailed_rows: list[dict],
                              conversation_rows: list[dict] | None) -> str:
    """Final section: Metric strengths, weaknesses, and interpretation guidance."""
    lines = [
        "## N. Metric Interpretation Guide\n",
        "This section summarises what each metric family captures and where it falls short, "
        "to help interpret the results above correctly.\n",
    ]

    # Collect which metric groups are actually present
    has_bleu      = col_populated(detailed_rows, "bleu_score")
    has_rouge     = col_populated(detailed_rows, "rouge_l_f1")
    has_meteor    = col_populated(detailed_rows, "meteor_score")
    has_bert      = col_populated(detailed_rows, "bertscore_f1")
    has_bleurt    = col_populated(detailed_rows, "bleurt_score")
    has_entity    = col_populated(detailed_rows, "entity_f1")
    has_behav     = col_populated(detailed_rows, "hedge_rate_predicted")
    has_judge     = col_populated(detailed_rows, "judge_mean")
    has_conv      = conversation_rows is not None and len(conversation_rows) > 0
    has_style     = col_populated(detailed_rows, "style_compliance_score")
    has_loss      = col_populated(detailed_rows, "loss")

    # ── Lexical metrics ──
    if has_bleu or has_rouge or has_meteor:
        lines.append("### Lexical Metrics (BLEU, ROUGE-L, METEOR)\n")

        strengths = []
        weaknesses = []
        strengths.append("Fast to compute, deterministic, widely reported — useful as baselines for comparability with other work")
        strengths.append("METEOR improves over BLEU by accounting for synonyms and stemming")
        weaknesses.append(
            "BLEU and ROUGE measure n-gram overlap only — a semantically correct paraphrase "
            "can score near zero if it uses different words"
        )
        weaknesses.append(
            "Particularly unreliable for short conversational turns (1–3 sentences) where "
            "word choice varies naturally"
        )
        weaknesses.append(
            "BLEU scores in this dataset may be inflated or deflated by `<eos>` token handling "
            "in the upstream pipeline"
        )

        lines.append("**Strengths:**\n")
        for s in strengths:
            lines.append(f"- {s}")
        lines.append("\n**Weaknesses:**\n")
        for w in weaknesses:
            lines.append(f"- {w}")
        lines.append(
            "\n**Recommendation:** Treat BLEU/ROUGE as secondary reference metrics. "
            "Do not use them as the primary basis for model selection decisions.\n"
        )

    # ── Semantic metrics ──
    if has_bert or has_bleurt:
        lines.append("### Semantic Metrics (BERTScore, BLEURT)\n")

        strengths = []
        weaknesses = []

        if has_bert:
            strengths.append(
                "BERTScore uses contextual embeddings to capture meaning-level similarity — "
                "correctly scores valid paraphrases higher than lexical metrics would"
            )
            weaknesses.append(
                "BERTScore requires a transformer model at inference time; "
                "scores vary depending on the underlying model (DeBERTa-xlarge-MNLI recommended)"
            )
            weaknesses.append(
                "Token-level greedy matching can be thrown off by very short texts "
                "where a single token mismatch dominates the score"
            )

        if has_bleurt:
            strengths.append(
                "BLEURT is trained on human quality judgments — highest correlation with human "
                "evaluation among automated metrics (Sellam et al., 2020)"
            )
            weaknesses.append(
                "BLEURT scores are not bounded [0,1] — negative values are valid and indicate "
                "below-reference quality; requires calibration to interpret absolute values"
            )

        lines.append("**Strengths:**\n")
        for s in strengths:
            lines.append(f"- {s}")
        lines.append("\n**Weaknesses:**\n")
        for w in weaknesses:
            lines.append(f"- {w}")
        lines.append(
            "\n**Recommendation:** Use BERTScore F1 as the primary automated metric for model "
            "selection. BLEURT provides a complementary learned-quality signal.\n"
        )

    # ── Entity metrics ──
    if has_entity:
        lines.append("### Entity Precision / Recall / F1\n")

        lines.append("**Strengths:**\n")
        lines.append(
            "- Directly measures factual fidelity — whether the model preserves amounts, dates, "
            "card numbers, and merchant names from the conversation context\n"
            "- Cheap to compute (regex-based), interpretable, and domain-specific\n"
        )
        lines.append("**Weaknesses:**\n")
        lines.append(
            "- Regex extraction is brittle — may miss entities in unusual formats or "
            "extract false positives from surrounding text\n"
            "- Only covers a fixed set of entity types; adding new types requires manual regex work\n"
            "- Many conversational turns contain zero entities, making the metric undefined "
            "(scored as 1.0 precision/recall by convention)\n"
        )
        lines.append(
            "**Recommendation:** Focus on turns where `entity_count_ref > 0`. "
            "Low entity recall is a strong signal of information loss; "
            "low precision signals hallucinated details.\n"
        )

    # ── Behavioural signals ──
    if has_behav:
        lines.append("### Behavioural Signals (Hedge Rate, Promise Rate, TTR, Role Confusion)\n")

        lines.append("**Strengths:**\n")
        lines.append(
            "- Captures pragmatic properties invisible to all other metrics — "
            "whether the model sounds like a *user* rather than an *assistant*\n"
            "- Role confusion detection directly flags the most dangerous failure mode "
            "for a UserLM: producing assistant-like language\n"
            "- Hedge and promise rate deltas are grounded in empirical findings from "
            "Wang et al. (2025) on real human–LLM behavioural divergences\n"
        )
        lines.append("**Weaknesses:**\n")
        lines.append(
            "- Phrase-list based detection (hedges, promises) has limited coverage — "
            "novel hedging strategies will be missed\n"
            "- TTR is sensitive to text length; short turns inflate TTR\n"
            "- Role confusion regex may flag legitimate user phrases that happen to overlap "
            "with assistant language (e.g., \"I can provide\" in a business-savvy persona)\n"
        )
        lines.append(
            "**Recommendation:** Role confusion rate is a hard diagnostic — any non-zero value "
            "warrants manual inspection. Hedge/promise deltas are directional indicators: "
            "systematic negative hedge Δ or positive promise Δ suggests the model has not "
            "fully learned human pragmatic patterns.\n"
        )

    # ── LLM-as-judge ──
    if has_judge:
        lines.append("### LLM-as-Judge\n")

        lines.append("**Strengths:**\n")
        lines.append(
            "- Most comprehensive single evaluation — scores across six dimensions that no "
            "automated metric can cover (persona voice, goal directedness, human realism)\n"
            "- Can incorporate conversation context, not just the single reference/prediction pair\n"
            "- Flexible rubric that can be extended or adjusted as requirements evolve\n"
        )
        lines.append("**Weaknesses:**\n")
        lines.append(
            "- Expensive (API cost per example) and slow compared to automated metrics\n"
            "- LLM judges can exhibit position bias, verbosity bias, and self-preference "
            "(tendency to rate LLM-generated text higher)\n"
            "- Scores are not deterministic — running the same example twice may yield "
            "different scores (mitigated by caching)\n"
            "- A single judge model may have systematic blind spots; ideally validated against "
            "human annotator agreement\n"
        )
        lines.append(
            "**Recommendation:** Use judge scores for qualitative model comparison and "
            "identifying dimensional weaknesses (e.g., all models weak on persona voice). "
            "Do not rely solely on judge mean for model ranking — cross-validate with "
            "BERTScore and behavioural metrics.\n"
        )

    # ── Conversation-level ──
    if has_conv:
        lines.append("### Conversation-Level Metrics (Self-BLEU, Trajectory Correlations)\n")

        lines.append("**Strengths:**\n")
        lines.append(
            "- Only metric family that captures cross-turn coherence — essential for evaluating "
            "a UserLM that must maintain persona consistency across a full conversation\n"
            "- Self-BLEU is a direct mode-collapse detector: high values mean the model "
            "is producing repetitive turns within a conversation\n"
            "- Trajectory correlations (hedge ρ, length ρ) measure whether the model tracks "
            "the natural escalation/de-escalation dynamics of a conversation\n"
        )
        lines.append("**Weaknesses:**\n")
        lines.append(
            "- Requires enough turns per conversation to be meaningful "
            "(conversations with ≤2 text turns produce unreliable correlations)\n"
            "- Spearman ρ can be undefined or misleading when all values in a trajectory are "
            "identical (e.g., hedge rate = 0 for all turns)\n"
            "- These metrics are computed against *synthetic reference conversations*, not "
            "real human conversations — the reference trajectory itself may not be realistic\n"
        )
        lines.append(
            "**Recommendation:** Focus on Self-BLEU for mode collapse detection "
            "and style compliance σ for consistency. Trajectory correlations are most "
            "informative for longer conversations (5+ turns).\n"
        )

    # ── Loss / Perplexity ──
    if has_loss:
        lines.append("### Loss & Perplexity\n")

        lines.append("**Strengths:**\n")
        lines.append(
            "- Direct training objective — the most granular signal of how well the model "
            "has learned the target distribution at the token level\n"
            "- Perplexity is comparable across model sizes when computed on the same data\n"
        )
        lines.append("**Weaknesses:**\n")
        lines.append(
            "- Low perplexity does not guarantee high-quality text — the model could have "
            "low perplexity by always predicting safe, generic responses\n"
            "- Perplexity values can be dominated by common tokens (articles, prepositions) "
            "rather than the information-carrying words\n"
            "- Very large perplexity values (>10,000) often indicate a computation bug "
            "(loss summed rather than averaged before exponentiation)\n"
        )
        lines.append(
            "**Recommendation:** Use loss curves for training diagnostics (overfitting detection) "
            "but not for model selection. Validate against generation-quality metrics above.\n"
        )

    # ── Style compliance ──
    if has_style:
        lines.append("### Style Compliance\n")

        lines.append("**Strengths:**\n")
        lines.append(
            "- Directly measures a core UserLM requirement: does the model match the "
            "persona's communication style (terse/direct/indirect)?\n"
            "- Cheap, deterministic, and interpretable\n"
        )
        lines.append("**Weaknesses:**\n")
        lines.append(
            "- Rule-based thresholds (e.g., terse = ≤12 words) are arbitrary and may not "
            "capture the full range of valid stylistic expression\n"
            "- Binary compliance misses degree — a 13-word 'terse' response is scored the "
            "same as a 50-word response\n"
        )
        lines.append(
            "**Recommendation:** Use alongside length ratio error for a more nuanced view. "
            "Style compliance is most useful for identifying categorical failures "
            "(e.g., indirect persona generating terse responses).\n"
        )

    return "\n".join(lines)


def section_loss(overall: dict[str, dict]) -> str:
    """Section 12: Loss & perplexity."""
    variants = variants_present(overall)
    lines = [
        "## 12. Loss & Perplexity\n",
        "Mean cross-entropy loss and perplexity across all examples (including EOS).\n",
        "| Model | Mean Loss | Mean Perplexity |",
        "|-------|-----------|----------------|",
    ]
    for v in variants:
        r = overall[v]
        lines.append(f"| {v} | {f(r.get('mean_loss'))} | {f(r.get('mean_perplexity'))} |")
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

    conversation_rows = None
    if CONVERSATION_CSV.exists():
        conversation_rows = load_csv(CONVERSATION_CSV)
        print(f"  → conversation_metrics.csv loaded ({len(conversation_rows)} rows)")

    overall = get_overall(category_rows)
    _, _, primary_name = detect_primary_metric(overall)
    print(f"  → Primary metric detected: {primary_name}")

    variants = variants_present(overall)
    n_examples = int(overall.get(variants[0], {}).get("n") or 0) if variants else 0
    n_eos      = int(overall.get(variants[0], {}).get("n_eos") or 0) if variants else 0
    n_models   = len(variants)
    n_lora     = len(lora_variants_present(overall))

    # Detect which metric groups are available
    available_metrics = []
    for cat_col, det_col, name in METRIC_HIERARCHY:
        if col_populated(detailed_rows, det_col):
            available_metrics.append(name)
    if col_populated(detailed_rows, "entity_f1"):
        available_metrics.append("Entity P/R/F1")
    if col_populated(detailed_rows, "hedge_rate_predicted"):
        available_metrics.append("Behavioural Signals")
    if col_populated(detailed_rows, "judge_mean"):
        available_metrics.append("LLM-as-Judge")
    if conversation_rows:
        available_metrics.append("Conversation-Level")

    print("Building report …")

    # Header
    sections = [
        "# Prediction Evaluation Report\n",
        f"**Dataset:** {n_examples} validation examples "
        f"({n_eos} EOS · {n_examples - n_eos} text)  \n"
        f"**Model variants evaluated:** {n_models} "
        f"({n_lora} LoRA fine-tuned + {n_models - n_lora} base)  \n"
        f"**Primary metric:** {primary_name}  \n"
        f"**Available metrics:** {' · '.join(available_metrics)}\n",
        "---\n",
    ]

    # Build sections — optional ones return None when data is missing
    section_fns = [
        lambda: section_overview(overall, category_rows),
        lambda: section_semantic(overall, detailed_rows),
        lambda: section_eos(category_rows, detailed_rows),
        lambda: section_behavioural(overall, detailed_rows),
        lambda: section_entity(overall, detailed_rows),
        lambda: section_judge(overall, detailed_rows),
        lambda: section_conversation(conversation_rows, overall),
        lambda: section_persona(category_rows, overall),
        lambda: section_model_comparison(comparison_rows, overall),
        lambda: section_efficiency(overall),
        lambda: section_style(category_rows, overall),
        lambda: section_loss(overall),
        lambda: section_metric_commentary(overall, detailed_rows, conversation_rows),
    ]

    section_num = 1
    for fn in section_fns:
        content = fn()
        if content is None:
            continue
        # Re-number sections dynamically
        import re
        content = re.sub(r"^## [\dN]+\.", f"## {section_num}.", content, count=1)
        sections.append(content)
        sections.append("\n---\n")
        section_num += 1

    report = "\n".join(sections)

    with open(REPORT_PATH, "w", encoding="utf-8") as f_out:
        f_out.write(report)

    print(f"  ✓  Report written to {REPORT_PATH}")
    print(f"  ✓  {section_num - 1} sections generated")


if __name__ == "__main__":
    main()
