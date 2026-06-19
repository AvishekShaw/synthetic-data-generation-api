# Conversation-Level Fidelity Evaluation Report

Evaluates whether reconstructed predicted conversations reproduce the human-realism properties of the reference conversations. Each conversation is reconstructed by ordering the predicted turns for a given `(model_variant, generation_idx)` pair and comparing them as a sequence against the equivalent reference turns.

**Conversations evaluated:** 39 unique conversations × 6 model variants = 234 total  
**Available tiers:** Tier 1 Rule-Based Fidelity · Tier 2 LLM Judge Fidelity  
**Source files:** `conversation_predictions_t1.csv` · `conversation_predictions_t2.csv` · `images/cp_*.png`

> **How to read this report:** Tier 1 metrics compare predicted vs reference turn sequences using rule-based checks. Δ values are (predicted − reference): values close to zero indicate high fidelity. Fidelity flags are raised when a dimension deviates beyond a calibrated threshold.

---

## 1. Conversation-Level Tier 1 — Rule-Based Fidelity

Rule-based metrics computed on reconstructed predicted conversations and their reference counterparts. Δ = predicted − reference. Flags are raised when the predicted conversation deviates meaningfully from reference behaviour on that dimension.

### A. Length & Lexical Deltas

| Model | ΔWords/turn | ΔHedge/turn | ΔCertainty/turn | ΔPromise/turn |
|-------|------------|------------|----------------|--------------|
| base_4b | +7.186 | -0.010 | +0.065 | +0.000 |
| lora_4b | -3.026 | -0.087 | -0.012 | +0.000 |
| base_12b | +8.401 | +0.021 | +0.088 | +0.005 |
| lora_12b | -2.579 | +0.004 | -0.005 | +0.000 |
| base_27b | +9.453 | +0.240 | +0.092 | +0.000 |
| lora_27b | -2.622 | +0.036 | +0.007 | +0.000 |

> Negative ΔWords = model generates shorter turns than reference. Negative ΔHedge = model under-hedges relative to reference (less tentative). Positive ΔCertainty = model over-commits relative to reference.

### B. Information Density

| Model | Pred InfoDensity | Ref InfoDensity | Δ | BF Pred % | BF Ref % |
|-------|----------------|----------------|---|----------|---------|
| base_4b | 0.333 | 0.026 | +0.308 | 5.1% | 5.1% |
| lora_4b | 0.000 | 0.026 | -0.026 | 2.6% | 5.1% |
| base_12b | 0.171 | 0.026 | +0.145 | 20.5% | 5.1% |
| lora_12b | 0.017 | 0.026 | -0.009 | 5.1% | 5.1% |
| base_27b | 0.385 | 0.026 | +0.359 | 7.7% | 5.1% |
| lora_27b | 0.000 | 0.026 | -0.026 | 2.6% | 5.1% |

> InfoDensity = fraction of key facts (merchant / amount / date) volunteered in the first customer turn. BF % = fraction of conversations where the model bundles 2+ concern categories in a single turn (breadth-first tell).

### C. Fidelity Flags

Fraction of conversations where the predicted sequence raises each flag. Lower = better fidelity. **MissedPushback** = type_b conversations where reference pushes back but prediction doesn't. **WrongBeliefMissing** = type_a conversations where reference expresses wrong prior belief but prediction doesn't. **DropoutImplausible** = failure conversations where the predicted exit lacks mode-appropriate dropout language (e.g. no impatience/escalation signals before leaving).

| Model | LenInflated | LenDeflated | OverCertain | UnderHedge | FrontLoad | BreadthNew | RoleConf | MissedPushback | WrongBeliefMissing | DropoutImplausible |
|---|---|---|---|---|---|---|---|---|---|---|
| base_4b | 82.1% | 2.6% | 41.0% | 33.3% | 59.0% | 2.6% | 12.8% | 50.0%* | 71.4%* | 100.0%* |
| lora_4b | 5.1% | 76.9% | 12.8% | 23.1% | 0.0% | 2.6% | 0.0% | 12.5%* | 57.1%* | 100.0%* |
| base_12b | 89.7% | 0.0% | 33.3% | 23.1% | 28.2% | 17.9% | 12.8% | 25.0%* | 57.1%* | 100.0%* |
| lora_12b | 12.8% | 61.5% | 20.5% | 15.4% | 2.6% | 5.1% | 0.0% | 37.5%* | 57.1%* | 100.0%* |
| base_27b | 94.9% | 0.0% | 41.0% | 7.7% | 53.8% | 7.7% | 2.6% | 50.0%* | 71.4%* | 60.0%* |
| lora_27b | 0.0% | 64.1% | 15.4% | 10.3% | 0.0% | 0.0% | 0.0% | 50.0%* | 71.4%* | 100.0%* |

> \* Type-specific flags computed over type-specific subset only (type_b for MissedPushback, type_a for WrongBeliefMissing, failure for DropoutImplausible).

### D. Type-Specific Signal Breakdown

Per-type averages for pushback, prior-belief, and dropout signals. Pushback = predicted pushback turns per conversation (type_b only). PriorBelief rate = fraction of predicted turns expressing the scenario's wrong belief (type_a only). DropoutTurn = mean turn index at which the predicted conversation exits (failure only).

| Model | Conv Type | N convs | Pred Pushback turns | Ref Pushback turns | Pushback missed | Pred PriorBelief rate | Pred DropoutTurn | Ref DropoutTurn | DropoutImplausible |
|---|---|---|---|---|---|---|---|---|---|
| base_4b | success | 19 | — | — | — | — | — | — | — |
| base_4b | failure | 5 | — | — | — | — | 5.400 | 5.400 | 100.0% |
| base_4b | type_a | 7 | — | — | — | 0.000 | — | — | — |
| base_4b | type_b | 8 | 0.000 | 0.875 | 50.0% | — | — | — | — |
| lora_4b | success | 19 | — | — | — | — | — | — | — |
| lora_4b | failure | 5 | — | — | — | — | 4.200 | 5.400 | 100.0% |
| lora_4b | type_a | 7 | — | — | — | 0.024 | — | — | — |
| lora_4b | type_b | 8 | 0.500 | 0.875 | 12.5% | — | — | — | — |
| base_12b | success | 19 | — | — | — | — | — | — | — |
| base_12b | failure | 5 | — | — | — | — | 5.400 | 5.400 | 100.0% |
| base_12b | type_a | 7 | — | — | — | 0.018 | — | — | — |
| base_12b | type_b | 8 | 0.500 | 0.875 | 25.0% | — | — | — | — |
| lora_12b | success | 19 | — | — | — | — | — | — | — |
| lora_12b | failure | 5 | — | — | — | — | 4.200 | 5.400 | 100.0% |
| lora_12b | type_a | 7 | — | — | — | 0.041 | — | — | — |
| lora_12b | type_b | 8 | 0.250 | 0.875 | 37.5% | — | — | — | — |
| base_27b | success | 19 | — | — | — | — | — | — | — |
| base_27b | failure | 5 | — | — | — | — | 5.400 | 5.400 | 60.0% |
| base_27b | type_a | 7 | — | — | — | 0.000 | — | — | — |
| base_27b | type_b | 8 | 0.000 | 0.875 | 50.0% | — | — | — | — |
| lora_27b | success | 19 | — | — | — | — | — | — | — |
| lora_27b | failure | 5 | — | — | — | — | 4.200 | 5.400 | 100.0% |
| lora_27b | type_a | 7 | — | — | — | 0.000 | — | — | — |
| lora_27b | type_b | 8 | 0.000 | 0.875 | 50.0% | — | — | — | — |

![cp_01_t1_overview.png](images/cp_01_t1_overview.png)

![cp_02_t1_deltas.png](images/cp_02_t1_deltas.png)

![cp_03_t1_flags.png](images/cp_03_t1_flags.png)

_cp_01: Predicted vs reference mean words/turn, hedge rate, certainty rate. cp_02: Delta bar charts (pred − ref) for key metrics. cp_03: Fidelity flag heatmap — darker = more flags raised._

![cp_06_t1_by_conv_type.png](images/cp_06_t1_by_conv_type.png)

_cp_06: Fidelity flag rates stratified by conversation type (success / failure / type_a / type_b). MissedPushback, WrongBeliefMissing, and DropoutImplausible appear only in their respective type columns._

---

## 2. Conversation-Level Tier 2 — LLM Judge Fidelity

LLM-as-judge scores measuring how faithfully the predicted conversations reproduce the human-realism properties of the reference conversations. Scored 1–5 per dimension: 1 = completely divergent, 5 = indistinguishable from reference. The judge evaluates FIDELITY, not absolute quality. Each conversation type is evaluated by a dedicated judge rubric with type-specific dimensions.

### Success (n=114)

| Model | Depth-First | Uncertainty | Info Drip | Pragmatic | Persona | OVERALL |
|---|---|---|---|---|---|---|
| base_4b | 1.947 | 2.211 | 1.895 | 1.474 | 2.105 | 1.897 |
| lora_4b | 3.474 | 3.579 | 2.684 | 3.789 | 3.158 | 3.408 |
| base_12b | 2.105 | 2.842 | 2.158 | 1.632 | 2.368 | 2.182 |
| lora_12b | 3.316 | 4.211 | 2.632 | 3.579 | 3.211 | 3.442 |
| base_27b | 2.579 | 2.421 | 1.895 | 2.895 | 2.684 | 2.539 |
| lora_27b | 3.474 | 4.579 | 3.158 | 3.947 | 3.474 | 3.766 |

> **LoRA weakest (Success):** Info Drip (2.82)  
> **LoRA strongest (Success):** Uncertainty (4.12)

### Failure — Dropout (n=30)

| Model | DropoutAuth | Pragmatic | Persona | Depth-First | Uncertainty | OVERALL |
|---|---|---|---|---|---|---|
| base_4b | 1.600 | 1.400 | 2.200 | 2.200 | 2.400 | 1.860 |
| lora_4b | 2.200 | 3.400 | 2.800 | 3.000 | 2.800 | 2.740 |
| base_12b | 1.000 | 1.000 | 1.800 | 1.600 | 2.800 | 1.480 |
| lora_12b | 2.000 | 3.600 | 2.800 | 3.200 | 3.200 | 2.800 |
| base_27b | 2.400 | 2.800 | 3.000 | 2.200 | 2.600 | 2.570 |
| lora_27b | 2.000 | 4.000 | 3.200 | 3.400 | 3.600 | 3.030 |

> **LoRA weakest (Failure — Dropout):** DropoutAuth (2.07)  
> **LoRA strongest (Failure — Dropout):** Pragmatic (3.67)

### Type A — Inadvertent Probing (n=42)

| Model | PriorBelief | InfoIncomplete | Depth-First | Uncertainty | Persona | OVERALL |
|---|---|---|---|---|---|---|
| base_4b | 1.000 | 2.429 | 2.286 | 2.000 | 2.571 | 1.929 |
| lora_4b | 1.429 | 3.857 | 3.429 | 3.429 | 3.000 | 2.850 |
| base_12b | 1.000 | 2.143 | 2.000 | 2.286 | 2.286 | 1.814 |
| lora_12b | 1.429 | 3.286 | 3.143 | 4.000 | 3.143 | 2.786 |
| base_27b | 1.286 | 2.000 | 2.143 | 2.143 | 2.857 | 1.964 |
| lora_27b | 1.857 | 3.429 | 3.000 | 3.714 | 3.286 | 2.893 |

> **LoRA weakest (Type A — Inadvertent Probing):** PriorBelief (1.57)  
> **LoRA strongest (Type A — Inadvertent Probing):** Uncertainty (3.71)

### Type B — Adversarial Probing (n=48)

| Model | ErrorDetect | PushbackCalib | Uncertainty | Persona | Pragmatic | OVERALL |
|---|---|---|---|---|---|---|
| base_4b | 1.500 | 2.125 | 2.250 | 2.250 | 1.750 | 1.906 |
| lora_4b | 2.250 | 2.625 | 2.875 | 2.750 | 3.625 | 2.650 |
| base_12b | 1.250 | 1.500 | 2.250 | 2.375 | 1.750 | 1.681 |
| lora_12b | 1.750 | 2.375 | 2.875 | 2.750 | 3.750 | 2.425 |
| base_27b | 1.250 | 2.125 | 2.500 | 2.625 | 2.625 | 2.000 |
| lora_27b | 2.125 | 2.375 | 3.125 | 2.750 | 3.750 | 2.594 |

> **LoRA weakest (Type B — Adversarial Probing):** ErrorDetect (2.04)  
> **LoRA strongest (Type B — Adversarial Probing):** Pragmatic (3.71)


![cp_04_t2_scores.png](images/cp_04_t2_scores.png)

![cp_05_t2_by_dimension.png](images/cp_05_t2_by_dimension.png)

![cp_07_t2_probe_scores.png](images/cp_07_t2_probe_scores.png)

_cp_07: Type-specific LLM judge scores for the three hardest behavioural contracts, evaluated only over the relevant conversation subset. Error Detection (type\_b, N=8): did the model catch and challenge the planted agent error? Prior Belief Persistence (type\_a, N=7): did the model maintain the wrong prior belief before correction? Dropout Authenticity (failure, N=5): did the exit feel earned with a plausible frustration arc? All three dimensions sit well below the midpoint (3) for all models, confirming these are the hardest contracts to reproduce and the primary targets for RL training._

---

## 3. Metric Interpretation Guide

This section covers what each metric family captures in the conversation-level report and where it has limitations.

### Tier 1 — Rule-Based Fidelity

**What it measures:** Five dimensions of conversation-level human realism, compared between predicted and reference turn sequences:
- **Length & lexical deltas** — whether the model generates turns of similar length and uses hedging/certainty language at the same rate as the reference
- **Information density** — whether the model front-loads key facts (merchant, amount, date) in the first turn at the same rate as the reference. High front-loading is an LLM tell (Wang et al. 2025)
- **Breadth-first bundling** — whether the model raises multiple concern categories (dispute + refund + card action) in the same turn, as LLMs tend to do
- **Persona signals** — whether emotional markers and banking jargon are present at the same rate as in the reference
- **Fidelity flags** — binary flags raised when a dimension deviates beyond a calibrated threshold; flag count is the primary fidelity health signal

**Strengths:** Fast, deterministic, interpretable, no API cost. Flag counts give a single at-a-glance fidelity health score per conversation.

**Weaknesses:** Phrase-list detection has limited coverage. Thresholds (±15% length, ±0.10 hedge rate) were calibrated for this dataset and may need adjustment as data scale grows. Metrics are computed on customer turns only — agent turns are not reconstructed.

**Recommendation:** Use flag counts as the primary Tier 1 signal. Investigate conversations with ≥3 flags manually. `flag_under_hedge` and `flag_role_confused` are the most diagnostic for UserLM quality.

### Tier 2 — LLM Judge Fidelity

**What it measures:** Five dimensions scored 1–5 by an LLM judge (claude-sonnet-4-6), framed as FIDELITY: does the predicted conversation reproduce the same human-realism properties as the reference? Each conversation type uses a dedicated judge rubric with type-specific dimensions and weights:

**Success conversations** (normal dispute resolution):
- **Depth-First Questioning** — does the model raise concerns sequentially, not bundled?
- **Uncertainty Expression** — does hedge/certainty balance match reference?
- **Information Drip** — does key info appear in the same turns as reference?
- **Pragmatic Naturalness** — do predicted turns sound as colloquial as reference?
- **Persona Fidelity** — is the assigned persona as visible as in reference?
- *Weights: Depth-First (0.25) · Pragmatic (0.25) · Uncertainty (0.20) · Info Drip (0.15) · Persona (0.15)*

**Failure — Dropout conversations** (customer exits before goal completion):
- **Dropout Authenticity** — does the model exit at a plausible point and with mode-appropriate exit language, matching the reference's dropout pattern?
- **Pragmatic Naturalness** — do the pre-exit turns sound as colloquial and frustrated as the reference, rather than ending abruptly?
- **Persona Fidelity**, **Depth-First Questioning**, **Uncertainty Expression** (same as success)
- *Weights: DropoutAuth (0.25) · Pragmatic (0.25) · Persona (0.15) · Depth-First (0.15) · Uncertainty (0.15) · Info Drip (0.05)*

**Type A — Inadvertent Probing** (user holds a wrong prior belief):
- **Prior Belief Persistence** — does the model maintain the wrong belief long enough before being corrected, as the reference does?
- **Info Incompleteness** — does the model withhold or omit information in the same pattern as the reference (partial info-drip)?
- **Depth-First Questioning**, **Uncertainty Expression**, **Persona Fidelity** (same as success)
- *Weights: PriorBelief (0.30) · Depth-First (0.20) · InfoIncomplete (0.20) · Uncertainty (0.15) · Persona (0.15)*

**Type B — Adversarial Probing** (agent makes a planted error, user should push back):
- **Error Detection** — does the model catch and challenge the planted agent error, as the reference does?
- **Pushback Calibration** — is the pushback proportionate and appropriately assertive?
- **Uncertainty Expression**, **Persona Fidelity**, **Pragmatic Naturalness** (same as success)
- *Weights: ErrorDetect (0.35) · PushbackCalib (0.25) · Uncertainty (0.15) · Persona (0.15) · Pragmatic (0.10)*

**Strengths:** Holistic, context-aware, captures properties no rule can detect. Rationale fields explain the score in terms of specific turns.

**Weaknesses:** Expensive (API cost per conversation). LLM judges can exhibit self-preference bias — may rate LLM-generated turns higher than a human annotator would. Scores are non-deterministic; cached to avoid variance between runs.

**Recommendation:** Use Tier 2 to explain Tier 1 flag patterns and to identify the weakest fidelity dimension across model variants. For failure conversations, pay particular attention to DropoutAuth — a model that ends abruptly without earned frustration build-up will score low here regardless of turn-level BERTScore. For type_b, pay particular attention to ErrorDetect — a model that never pushes back will have high turn-level BERTScore but fail this dimension completely. Do not use Tier 2 overall score alone for model selection — cross-validate with Tier 1 flags and turn-level BERTScore.


---
