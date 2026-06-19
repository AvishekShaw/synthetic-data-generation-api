# Synthetic Conversation Evaluation — Metrics Reference

---

## eval_conversations.py — Layer 1 (rule-based, no API)

All metrics are computed from customer turns only. Results written to `eval_results.csv`.

### A. Structural Stats

| Metric | What it measures | How it's computed |
|---|---|---|
| `num_customer_turns` | Number of customer turns in the conversation | Count of turns where `speaker == "Customer"` |
| `words_mean` | Average customer turn length | `sum(word_counts) / num_customer_turns` |
| `words_median` | Median customer turn length | 50th percentile of per-turn word counts |
| `words_p10` / `words_p90` | Shortest / longest end of turn distribution | 10th and 90th percentile of per-turn word counts |
| `words_stddev` | Turn-length variance within a conversation | Population stddev of per-turn word counts |
| `HIGH_AVG_TURN_LENGTH` flag | LLMs over-explain; real users are brief | Raised when `words_mean > 30` |
| `LOW_LENGTH_VARIANCE` flag | Unnaturally uniform turn rhythm | Raised when `words_stddev < 3.0` with ≥3 turns |

### B. Lexical Signals

Grounded in Wang et al. (2025): LLM-generated users over-use certainty language; real users hedge.

| Metric | What it measures | How it's computed |
|---|---|---|
| `hedge_rate` | Hedging phrases per customer turn | Count of matches from `HEDGE_PHRASES` list (22 phrases, e.g. "I think", "maybe", "not sure") divided by `num_customer_turns` |
| `certainty_rate` | Over-certainty phrases per customer turn | Count of matches from `CERTAINTY_PHRASES` list (11 phrases, e.g. "definitely", "absolutely") divided by `num_customer_turns` |
| `hedge_minus_certainty` | Net hedging signal; positive = more human-like | `hedge_rate − certainty_rate` |
| `CERTAINTY_EXCEEDS_HEDGING` flag | Uncertain persona sounds over-confident | Raised when `certainty_rate > hedge_rate` for `calm` / `mildly_frustrated` personas or `certainty == uncertain` scenarios |
| `ZERO_HEDGING_FOR_UNCERTAIN_SCENARIO` flag | Uncertain scenario but no hedging at all | Raised when `certainty == uncertain` and total hedge count is 0 |

### C. First-Turn Info Density

| Metric | What it measures | How it's computed |
|---|---|---|
| `facts_in_turn1` | How many key facts the customer volunteers unprompted | Count of {merchant, amount, date} that appear in the customer's first turn (0–3). Amount matched on digit substring; date matched on day number OR month name |
| `info_density_score` | Normalised info-dump score | `facts_in_turn1 / 3` |
| `INFO_DUMP_TURN1` flag | LLM-like front-loading of all scenario facts | Raised when `facts_in_turn1 == 3` |

### D. Breadth-First Topic Bundling

| Metric | What it measures | How it's computed |
|---|---|---|
| `max_categories_per_turn` | Most concern categories in a single turn | For each customer turn, check matches across 4 categories (dispute, card_action, refund_credit, timeline_status); record the maximum |
| `turns_with_multi_category` | Turns bundling 2+ concern categories | Count of customer turns with ≥2 category hits |
| `breadth_first_flag` | Conversation-level LLM-like bundling signal | `True` when `max_categories_per_turn ≥ 3` OR `turns_with_multi_category ≥ 2` |
| `BREADTH_FIRST_BUNDLING` warning | Explicit flag raised on the same condition | Included in `warnings` list with counts as context |

### E. Persona Consistency

#### E1 — Emotional State

| Warning | What it checks | Condition |
|---|---|---|
| `MISSING_FRUSTRATION_MARKERS` | `mildly_frustrated` persona shows no frustration | Neither mild nor strong frustration markers found |
| `UNEXPECTED_FRUSTRATION` | `calm` persona shows frustration | Any mild or strong marker found in customer text |
| `MISSING_STRONG_FRUSTRATION` | `very_upset` / `escalating` persona lacks intensity | No matches from `STRONG_FRUSTRATION_MARKERS` (20 phrases, e.g. "unacceptable", "cfpb", "close my account") |

Mild markers (11 phrases): "look", "already said", "come on", "seriously", "all i want", etc.

#### E2 — Knowledge Level

| Metric | What it measures | How it's computed |
|---|---|---|
| `jargon_before_agent` | Customer uses specialist terms before agent introduces them | Walk turns in order; track which of 9 `BANKING_JARGON` terms the agent has said; flag any customer turn that uses an unseen term |
| `NOVICE_JARGON` warning | Novice using domain vocabulary | Raised when `knowledge_level == novice` and `jargon_before_agent == True` |

Banking jargon tracked: provisional credit, chargeback, reg e, regulation e, billing error, fraud claim, dispute resolution, zero liability, fcba.

#### E3 — Communication Style

| Warning | Condition |
|---|---|
| `DIRECT_STYLE_BUT_LONG_TURNS` | `communication_style == direct` and `words_mean > 25` |
| `INDIRECT_STYLE_BUT_VERY_SHORT` | `communication_style == indirect` and `words_mean < 6` |
| `POPULATION_STYLE_MISMATCH` (population) | Across all conversations, mean turn length for `direct` ≥ mean for `indirect` |

### F. Type-Specific Checks

#### F1 — Failure / Dropout Conversations

| Warning | What it checks | Condition |
|---|---|---|
| `DROPOUT_TOO_EARLY` | Authentic dropout requires build-up | `num_customer_turns < 3` before `<eos>` |
| `DROPOUT_NOT_PLAUSIBLE` | Final turn lacks mode-appropriate exit language | Last customer turn has no matches from `DROPOUT_SIGNALS[failure_mode]` (7 modes, 7–13 signals each) |
| `MISSING_ESCALATION_ARC` | Impatience/escalation exits need tone deterioration | `failure_mode in (impatience, escalation_exit)` and no frustration markers in the second half of the conversation (requires ≥4 turns) |

#### F2 — Type A (Inadvertent Probing / Wrong Prior Belief)

| Warning | What it checks | Condition |
|---|---|---|
| `PRIOR_BELIEF_ABSENT` | Wrong prior belief should surface in customer turns | Fewer than 2 keyword matches from `wrong_prior_belief` field found across all customer turns |
| `BELIEF_SILENTLY_DROPPED` | Prior belief should persist or be explicitly resolved | Keywords appear in first half but have zero matches in second half (requires ≥4 turns) |

#### F3 — Type B (Adversarial Probing / Planted Agent Error)

| Warning / Metric | What it checks | Condition |
|---|---|---|
| `ERROR_NOT_CAUGHT` | Customer should challenge the planted error | No matches from `PUSHBACK_PHRASES` (23 phrases) across all customer turns |
| `USER_CAUGHT_ERROR_LABEL_MISMATCH` | `_meta` label agrees with transcript | `user_caught_error == True` but no pushback found, or `user_caught_error == False` but pushback is present |

### Outlier Threshold

A conversation is flagged as `is_outlier = True` when `warning_count >= 3`.

---

## eval_layer2.py — Layer 2 (LLM-as-judge via Claude API)

Uses an LLM judge to score each conversation on behavioural dimensions. Scores are 1–5 where **1 = clearly LLM-like** and **5 = clearly human-like**. Results written to `eval_results_layer2.csv` and cached per-conversation in `layer2_cache/`.

Grounded in: Wang et al. (2025), Mannekote et al. (2025), Chen et al. (2024).

### Core Dimensions (all conversation types)

| Dimension | Score key | What it measures | Weight (success) |
|---|---|---|---|
| Depth-First Questioning | `depth_first_score` | Customer resolves one concern before raising another; no breadth-first bundling | 0.25 |
| Appropriate Uncertainty | `uncertainty_score` | Hedging language matches scenario certainty; uncertain scenarios should produce tentative phrasing, not commitment language | 0.20 |
| Natural Info Drip | `info_drip_score` | Customer reveals merchant / amount / date gradually when prompted, not front-loaded in turn 1 | 0.15 |
| Pragmatic Naturalness | `pragmatic_score` | Every turn sounds like a real caller; colloquial phrasing; no scripted or template-like turns | 0.25 |
| Persona Fidelity | `persona_score` | Assigned emotional state, knowledge level, and communication style clearly manifest across the conversation | 0.15 |

### Type-Specific Dimensions

| Type | Dimension | Score key | What it measures | Weight |
|---|---|---|---|---|
| failure | Dropout Authenticity | `dropout_authenticity_score` | Dropout is grounded in what the agent said or failed to do; exit feels inevitable; language matches `failure_mode` | 0.25 |
| type_a | Prior Belief Coherence | `prior_belief_coherence_score` | Wrong prior belief surfaces in customer questions, leaves a footprint across multiple turns, and is either corrected or coherently persists | 0.15 |
| type_b | Error Catch Quality | `error_catch_quality_score` | Customer specifically challenges the correct planted error with a precise reference to the incorrect fact | 0.15 |
| type_b | Pushback Naturalness | `pushback_naturalness_score` | Pushback sounds tentative and real (not policy-reciting); tone matches assigned `emotional_state` | 0.15 |

### Dimension Weights by Dataset Type

| Dimension | success | failure | type_a | type_b |
|---|---|---|---|---|
| depth_first | 0.25 | 0.15 | 0.20 | 0.10 |
| uncertainty | 0.20 | 0.15 | 0.20 | 0.15 |
| info_drip | 0.15 | 0.05 | 0.10 | 0.10 |
| pragmatic | 0.25 | 0.25 | 0.20 | 0.20 |
| persona | 0.15 | 0.15 | 0.15 | 0.15 |
| dropout_authenticity | — | 0.25 | — | — |
| prior_belief_coherence | — | — | 0.15 | — |
| error_catch_quality | — | — | — | 0.15 |
| pushback_naturalness | — | — | — | 0.15 |

### Overall Score

`overall_score = Σ (dimension_score × weight)` for all applicable dimensions per conversation type. Range: 1.0–5.0 (weighted).

### Qualitative Fields

| Field | What it captures |
|---|---|
| `standout_issue` | Single most human-unrealistic element in the conversation |
| `standout_quality` | Single most realistic element |

### Layer 1 ↔ Layer 2 Correlation Plots (Fig 10)

Four scatter plots validating whether Layer 1 heuristics predicted the judge's scores:

| Layer 1 signal | Layer 2 dimension |
|---|---|
| `hedge_minus_certainty` | `uncertainty_score` |
| `info_density_score` | `info_drip_score` |
| `breadth_first_flag` | `depth_first_score` |
| `words_mean` | `pragmatic_score` |
