# Prediction Evaluation Report

**Dataset:** 139 validation examples (16 EOS · 123 text)  
**Model variants evaluated:** 6 (3 LoRA fine-tuned + 3 base)  
**Metrics:** BLEU · ROUGE-L · METEOR · Exact Match · EOS Recall · Style Compliance · Loss · Perplexity

---

## 1. Overall Performance

Core text-generation metrics averaged across all examples. BLEU, ROUGE-L F1, and METEOR are computed on non-EOS examples only (EOS examples have no text to compare).

| Model | N | BLEU | ROUGE-L F1 | METEOR | Exact Match | EOS Recall | Style Compliance |
|-------|---|------|------------|--------|-------------|------------|-----------------|
| base_4b | 139 | 0.154 | 0.277 | 0.941 | 81.3% | 100.0% | 0.836 |
| lora_4b | 139 | 0.735 | 0.848 | 0.955 | 89.9% | 100.0% | 0.836 |
| base_12b | 139 | 0.144 | 0.281 | 0.932 | 76.3% | 100.0% | 0.834 |
| lora_12b | 139 | 0.745 | 0.841 | 0.965 | 97.8% | 100.0% | 0.836 |
| base_27b | 139 | 0.156 | 0.272 | 0.911 | 66.9% | 100.0% | 0.834 |
| lora_27b | 139 | 0.733 | 0.846 | 0.960 | 96.4% | 100.0% | 0.836 |

> **Best LoRA variant by BLEU:** `lora_12b` (BLEU 0.745, ROUGE-L 0.841, METEOR 0.965)

---

## 2. EOS Prediction Analysis

EOS examples are turns where the expected model output is `<eos>` — meaning the conversation should end. The correct prediction is an empty string `""`.

| Model | EOS examples | EOS Recall | False Positive Rate |
|-------|-------------|------------|---------------------|
| base_4b | 16 | 100.0% | 0.0% |
| lora_4b | 16 | 100.0% | 0.0% |
| base_12b | 16 | 100.0% | 0.0% |
| lora_12b | 16 | 100.0% | 0.0% |
| base_27b | 16 | 100.0% | 0.0% |
| lora_27b | 16 | 100.0% | 0.0% |

> All models correctly predicted EOS for every EOS example.

![15_eos_analysis.png](images/15_eos_analysis.png)

_Panel A: EOS recall per model (fraction of EOS examples correctly predicted as empty). Panel B: false positive rate (fraction of text examples wrongly predicted as empty). Panel C: word-count distribution of predictions on EOS-expected examples — 0 words is correct._

---

## 3. Performance by Persona Category

BLEU scores broken down by persona attribute (LoRA variants, non-EOS examples only).

### Communication Style

| Communication Style | lora_4b | lora_12b | lora_27b |
|---|---|---|---|
| direct | 0.737 | 0.748 | 0.742 |
| indirect | 0.761 | 0.756 | 0.730 |
| terse | 0.724 | 0.734 | 0.707 |

> Easiest: **indirect** (avg BLEU 0.749)  · Hardest: **terse** (avg BLEU 0.722)

### Emotional State

| Emotional State | lora_4b | lora_12b | lora_27b |
|---|---|---|---|
| calm | 0.747 | 0.730 | 0.734 |
| escalating | 0.739 | 0.742 | 0.751 |
| mildly_frustrated | 0.729 | 0.752 | 0.730 |

> Easiest: **escalating** (avg BLEU 0.744)  · Hardest: **calm** (avg BLEU 0.737)

### Knowledge Level

| Knowledge Level | lora_4b | lora_12b | lora_27b |
|---|---|---|---|
| expert | 0.734 | 0.747 | 0.752 |
| intermediate | 0.754 | 0.740 | 0.721 |
| novice | 0.656 | 0.764 | 0.741 |

> Easiest: **expert** (avg BLEU 0.744)  · Hardest: **novice** (avg BLEU 0.720)

### Goal Clarity

| Goal Clarity | lora_4b | lora_12b | lora_27b |
|---|---|---|---|
| clear | 0.715 | 0.746 | 0.733 |
| vague | 0.779 | 0.741 | 0.732 |
| wrong_mental_model | 0.761 | 0.756 | 0.730 |

> Easiest: **vague** (avg BLEU 0.750)  · Hardest: **clear** (avg BLEU 0.731)


![13_category_heatmap.png](images/13_category_heatmap.png)

_Heatmap of mean BLEU per persona attribute value × LoRA model variant. Green = high BLEU, red = low. Reveals which persona types each model size handles well or struggles with._

---

## 4. Cross-Model Comparison

Pairwise win rates and Wilcoxon signed-rank test significance (paired, non-parametric). Win rate = fraction of examples where model A has higher BLEU than model B.

| Model A | Model B | A win rate | B win rate | Tie rate | Δ BLEU (A−B) | p-value | Significant |
|---------|---------|-----------|-----------|----------|------------|---------|------------|
| lora_4b | lora_12b | 40.3% | 48.2% | 11.5% | -0.0087 | 0.4837 | ns |
| lora_4b | lora_27b | 42.4% | 46.0% | 11.5% | +0.0023 | 0.8390 | ns |
| lora_12b | lora_27b | 46.8% | 41.7% | 11.5% | +0.0110 | 0.3985 | ns |
| base_4b | lora_4b | 0.0% | 88.5% | 11.5% | -0.5141 | 0.0000 | *** |
| base_4b | base_12b | 46.8% | 41.7% | 11.5% | +0.0094 | 0.4006 | ns |
| base_4b | lora_12b | 0.0% | 88.5% | 11.5% | -0.5228 | 0.0000 | *** |
| base_4b | base_27b | 46.0% | 42.4% | 11.5% | -0.0015 | 0.9829 | ns |
| base_4b | lora_27b | 0.0% | 88.5% | 11.5% | -0.5118 | 0.0000 | *** |
| lora_4b | base_12b | 88.5% | 0.0% | 11.5% | +0.5235 | 0.0000 | *** |
| lora_4b | base_27b | 88.5% | 0.0% | 11.5% | +0.5126 | 0.0000 | *** |
| base_12b | lora_12b | 0.0% | 88.5% | 11.5% | -0.5322 | 0.0000 | *** |
| base_12b | base_27b | 43.9% | 44.6% | 11.5% | -0.0109 | 0.3312 | ns |
| base_12b | lora_27b | 0.0% | 88.5% | 11.5% | -0.5212 | 0.0000 | *** |
| lora_12b | base_27b | 88.5% | 0.0% | 11.5% | +0.5213 | 0.0000 | *** |
| base_27b | lora_27b | 0.0% | 88.5% | 11.5% | -0.5103 | 0.0000 | *** |

![14_model_comparison.png](images/14_model_comparison.png)

_Left: pairwise win-rate matrix for LoRA variants (diagonal = mean BLEU). Right: mean BLEU difference with Wilcoxon significance markers (`*` p<0.05, `**` p<0.01, `***` p<0.001, `ns` = not significant)._

---

## 5. Efficiency Frontier

Mean BLEU / ROUGE-L vs model scale (parameters). Bubble size reflects mean generation time.

| Model | Params (B) | Mean BLEU | Mean ROUGE-L | Mean gen time (s) |
|-------|-----------|-----------|-------------|-------------------|
| base_4b | 4 | 0.154 | 0.277 | 3.995 |
| lora_4b | 4 | 0.735 | 0.848 | 2.758 |
| base_12b | 12 | 0.144 | 0.281 | 3.959 |
| lora_12b | 12 | 0.745 | 0.841 | 2.724 |
| base_27b | 27 | 0.156 | 0.272 | 4.119 |
| lora_27b | 27 | 0.733 | 0.846 | 2.786 |

> `lora_4b` outperforms `base_4b` by 0.5810 BLEU points.

> `lora_12b` outperforms `base_12b` by 0.6014 BLEU points.

> `lora_27b` outperforms `base_27b` by 0.5767 BLEU points.

![16_efficiency_frontier.png](images/16_efficiency_frontier.png)

_Scatter plots of mean BLEU (left) and ROUGE-L F1 (right) against model size. Triangles = LoRA fine-tuned, circles = base. Dashed/dotted lines are trend lines for each variant type._

---

## 6. Style Compliance

Rule-based check of whether predictions match the persona's communication style. Terse = ≤12 words, Direct = ≤20 words, Indirect = ≥8 words with hedging language. Score 0–1 (higher = more compliant).

| Style | lora_4b | lora_12b | lora_27b |
|-------|---|---|---|
| terse | 0.906 | 0.906 | 0.906 |
| direct | 0.817 | 0.817 | 0.817 |
| indirect | 0.775 | 0.775 | 0.775 |

> Hardest style to comply with: **indirect** (avg score 0.775)  · Easiest: **terse** (avg score 0.906)

![17_style_compliance.png](images/17_style_compliance.png)

_Left: grouped bar chart of style compliance score per communication style × model. Right: box plots of length ratio error per style — 0 means the prediction matched the expected token count exactly._

---

## 7. Loss & Perplexity

Mean cross-entropy loss and perplexity across all examples (including EOS).

| Model | Mean Loss | Mean Perplexity |
|-------|-----------|----------------|
| base_4b | 2.865 | 179.248 |
| lora_4b | 0.463 | 3.068 |
| base_12b | 2.777 | 166.819 |
| lora_12b | 0.446 | 2.997 |
| base_27b | 2.812 | 172.665 |
| lora_27b | 0.449 | 2.987 |