# UserLM Data Pipeline — Generation, Splitting, and Evaluation

**Project:** Fine-tuning Gemma 3 27B to simulate the user side of banking fraud dispute conversations
**Scripts:** `synthetic_data_generation/` · `evaluate_synthetic_conversations/`
**Full eval report:** `evaluate_synthetic_conversations/evaluation_report.md`

---

## Overview

The training data consists of four dataset types, each targeting a different behavioural profile. All four are generated synthetically via the Anthropic API, combined into a single JSONL, and split 80:20 conversation-wise before training.

| Type | Script | Conversations | generation_idx range |
|---|---|---|---|
| Success | `generate_conversations.py --mode success` | 96 (12 personas × 8 scenarios) | 0–499 |
| Failure / dropout | `generate_conversations.py --mode failure` | 40 | 500–999 |
| Type A — inadvertent probing | `generate_probe_conversations.py --mode type_a` | 96 | 1000–1095 |
| Type B — adversarial probing | `generate_probe_conversations.py --mode type_b` | 96 | 2000–2095 |

The `generation_idx` offset space is non-overlapping by design so that `split_dataset.py` can combine all four files and detect collisions.

---

## Dataset 1 — Success Conversations

**Script:** `generate_conversations.py --mode success`
**Architecture:** Single API call per conversation — the entire customer-side conversation is generated in one shot.

The generator creates a cross-product of 12 personas × 8 scenarios = 96 combinations. Each persona is defined by four attributes (knowledge level, emotional state, communication style, goal clarity); each scenario by six attributes (certainty, information completeness, prior contact, expected resolution, merchant, amount, transaction date).

A few-shot block of up to 3 real conversations is injected into the prompt from `--data-path` to communicate the expected format. On the first run with no prior data, `--data-path` can be omitted and the model generates cold.

Each generated conversation ends with `<eos>` on the final customer turn to signal goal completion. The output is a JSONL where each entry contains:
- `input`: either a `FIRST_TURN_TEMPLATE` (for the opening turn) or a `COMPLETION_TEMPLATE` with conversation history
- `output`: the next customer message + `<eos>`, or just `<eos>` for the final turn
- `_meta`: persona, scenario, `generation_idx`, `goal_completed: true`

**Why this matters for training:** These are the baseline examples that teach the model the fundamental shape of a banking dispute conversation from the user's side.

---

## Dataset 2 — Failure / Dropout Conversations

**Script:** `generate_conversations.py --mode failure --limit 40`
**Architecture:** Same single-call architecture as success, but with a different system prompt.

The failure system prompt instructs the model to generate a customer who exits before the goal is resolved, using a specific dropout trigger. The trigger is selected via `PERSONA_FAILURE_MAP` — a hardcoded mapping from persona index to failure mode so that each mode is represented across the dataset:

| Failure mode | What causes the exit |
|---|---|
| `impatience` | User runs out of patience and leaves |
| `info_blocker` | User doesn't have required information (card number, date) and can't proceed |
| `loop_exit` | Agent keeps asking the same question; user gives up |
| `trust_breakdown` | User loses confidence in the agent and stops engaging |
| `escalation_exit` | User demands a supervisor and leaves when that doesn't happen |
| `wrong_channel` | User realises this chat can't resolve their issue and drops off |
| `scope_limit` | Agent can't fulfill the request; user accepts defeat |
| `distraction` | User simply gets distracted mid-conversation |

The `FAILURE_COMPLETION_TEMPLATE` replaces the standard `COMPLETION_TEMPLATE` in these entries — it instructs the model to produce `<eos>` only when the user has decided to give up, not when the goal is complete. The `_meta` block records `goal_completed: false` and `failure_mode`.

**Target ratio:** ~40 failure conversations against ~96 success = roughly 1:2.5 failure:success. Keeping this below 33% prevents the model from learning that giving up is the default outcome.

**Why this matters for training:** Without dropout examples, `<eos>` means only one thing to the model — goal complete. Failure conversations teach it that `<eos>` can also mean the user walked away, and that the context preceding the exit (mounting frustration, repeated info requests, wrong-channel recognition) is what distinguishes the two.

---

## Dataset 3 — Type A: Inadvertent Probing

**Script:** `generate_probe_conversations.py --mode type_a`
**Architecture:** Alternating API calls — separate calls for each user turn and each agent turn, each with their own system prompt. This gives clean independent control over both sides of the conversation.

Three stress vectors are layered on top of each persona:

**1. Wrong prior belief** — each of the 8 scenarios has a specific misconception pre-loaded into the user system prompt:

| Scenario | Injected wrong belief |
|---|---|
| 0 | Dispute window is 120 days (correct: 90) |
| 1 | Can dispute any charge with just a rough description |
| 2 | Since the merchant already refused, the bank is legally required to refund immediately |
| 3 | Must be 100% sure it's fraud before the bank can do anything |
| 4 | Card freeze happens instantly and affects all pending transactions |
| 5 | Entitled to speak to a supervisor immediately upon request |
| 6 | Federal law requires provisional credit within 24 hours |
| 7 | Must file a police report before the bank can open a dispute |

**2. Information drip** — the user is instructed to reveal transaction details (merchant, amount, date) incrementally only when prompted, not upfront.

**3. Error awareness** — the user is instructed to notice if the agent says something inconsistent and gently push back.

The `_meta` block records `probing_type: inadvertent`, `goal_completed`, and `wrong_prior_belief`.

**Why this matters for training:** Type A captures realistic users who are confused or misinformed — not adversarial, just wrong. The model must learn to navigate belief correction gracefully, and to handle users who drip information rather than front-loading it.

---

## Dataset 4 — Type B: Adversarial Probing

**Script:** `generate_probe_conversations.py --mode type_b`
**Architecture:** Same alternating API call architecture as Type A.

The agent is given a system prompt that instructs it to make a specific planted error at a specific point in the conversation. The user is separately instructed to push back on agent errors proportionally to their persona. Each scenario has a distinct error type:

| Scenario | Planted error | Error type |
|---|---|---|
| 0 | States dispute window is 60 days (correct: 90) | `wrong_policy_fact` |
| 1 | Refuses to open dispute without exact merchant name (incorrect) | `scope_failure` |
| 2 | Promises refund in 24 hours (correct: 3–5 business days) | `over_commit` |
| 3 | States uncertain charges cannot be disputed (incorrect) | `wrong_policy_fact` |
| 4 | First says card freeze is instant, then says 24 hours on turn 4+ | `contradiction` |
| 5 | Asks for card/account number again after user already provided it | `loop` |
| 6 | Claims disputes over $1,000 require an in-branch visit (incorrect) | `wrong_escalation` |
| 7 | States a police report is required to open a dispute (incorrect — optional) | `wrong_policy_fact` |

A `user_caught_error` flag is set in `_meta` if the customer's turns contain pushback language (checked by scanning for phrases like "are you sure", "I thought", "hold on", "that doesn't sound right"). The `_meta` block also records `probing_type: adversarial`, `agent_failure_mode`, and `planted_error`.

**Why this matters for training:** This is the hardest behavioural target. The model must learn to challenge incorrect information rather than accept it passively. If the customer silently accepts a planted error, the training signal goes in the wrong direction entirely.

---

## Splitting the Dataset

**Script:** `split_dataset.py`

All four JSONL files are passed to a single splitter that operates at the conversation level (by `generation_idx`) rather than the entry level. This prevents leakage where turns from the same conversation end up in both train and val.

```bash
python split_dataset.py \
  --files userlm_success.jsonl userlm_failure.jsonl \
          userlm_type_a.jsonl userlm_type_b.jsonl \
  --ratio 0.8 --seed 42 \
  --train-out train_combined.jsonl \
  --val-out val_combined.jsonl
```

The script prints a breakdown by type, failure mode, goal completion, and user_caught_error before writing. Train entries are shuffled; val entries preserve conversation order. A collision guard raises an error if the same `generation_idx` appears in two input files.

**Combined dataset (success + failure only, current):** 120 conversations, 1032 entries — 817 train / 215 val.

---

## Evaluation Design

Evaluation runs in two complementary layers. Both scripts accept `--dumps-dir` to point at any conversation dump folder and auto-detect dataset type from the `EXTRA META` block. No manual flag is needed.

```bash
python eval_conversations.py --dumps-dir ../synthetic_data_generation/conversation_dumps_failure
python eval_layer2.py        --dumps-dir ../synthetic_data_generation/conversation_dumps_failure
```

### Layer 1 — Rule-Based Heuristic Checks (Sections A–F)

Fast, zero-cost, reproducible. Sections A–E apply to all types; Section F is type-gated.

**Sections A–E (all types)**

| Check | What it catches |
|---|---|
| `CERTAINTY_EXCEEDS_HEDGING` | Customer too confident — generators under-hedge relative to real humans |
| `BREADTH_FIRST_BUNDLING` | Multiple concern categories bundled in one turn — real callers raise one issue at a time |
| `INFO_DUMP_TURN1` | All three key facts (merchant, amount, date) front-loaded in the opening turn |
| `LOW_LENGTH_VARIANCE` | Turn lengths too uniform (σ < 3.0 words) |
| `UNEXPECTED_FRUSTRATION` | Frustration markers in calm-persona conversations |
| `MISSING_FRUSTRATION_MARKERS` | No frustration language in mildly-frustrated or escalating personas |

**Section F — type-specific**

*Failure:*

| Check | What it catches |
|---|---|
| `DROPOUT_TOO_EARLY` | Exit within the first 2 customer turns — real dropout requires build-up |
| `DROPOUT_NOT_PLAUSIBLE` | Final turn has no exit language consistent with the assigned `failure_mode` |
| `MISSING_ESCALATION_ARC` | For `impatience`/`escalation_exit` — no frustration in the second half of the conversation |

*Type A:*

| Check | What it catches |
|---|---|
| `PRIOR_BELIEF_ABSENT` | Keywords from the wrong prior belief appear in fewer than 2 customer turns |
| `BELIEF_SILENTLY_DROPPED` | Belief keywords appear in first half but vanish completely in second half with no resolution |

*Type B:*

| Check | What it catches |
|---|---|
| `ERROR_NOT_CAUGHT` | No pushback phrases anywhere — customer accepted the planted error without challenge |
| `USER_CAUGHT_ERROR_LABEL_MISMATCH` | `user_caught_error` flag in `_meta` contradicts what the transcript shows |

### Layer 2 — LLM-as-Judge

Five core dimensions scored 1–5 (1 = LLM-like, 5 = human-like), plus type-specific dimensions appended based on dataset type.

**Core dimensions (all types)**

| Dimension | What it measures |
|---|---|
| **Depth-first** | Does the customer resolve one issue before raising the next? |
| **Uncertainty** | Does hedging match the scenario's certainty level? |
| **Info drip** | Does the customer reveal details incrementally? |
| **Pragmatic** | Do turns read colloquial and natural rather than scripted? |
| **Persona** | Do emotional state, communication style, and knowledge level surface consistently? |

**Type-specific dimensions**

| Type | Dimension | What it measures |
|---|---|---|
| failure | **Dropout authenticity** | Is the exit grounded in what the agent said or failed to do? |
| type_a | **Prior belief coherence** | Does the wrong belief shape the conversation throughout, or appear once and vanish? |
| type_b | **Error catch quality** | Did the customer catch the *right* planted error specifically? |
| type_b | **Pushback naturalness** | Does the pushback sound like a real person noticing something off, or a policy-reciting bot? |

**Dimension weights by type**

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

For type_b, the depth_first rubric is explicitly relaxed in the prompt — adversarial conversations legitimately have overlapping concerns because the customer is simultaneously pursuing their goal and contesting incorrect information.

---

## What We Found (80 Success Conversations — Baseline)

**Strengths:**
- Pragmatic register is the strongest dimension (mean = 4.31) — conversations read naturally
- Persona traits surface correctly in ~85–90% of conversations
- `<eos>` termination is perfect — no false exits
- 52.5% of conversations score ≥ 4.5 overall

**Persistent weaknesses:**
- **Over-certainty** (30% flagged): generators under-hedge in fraud-certain scenarios
- **Breadth-first bundling** (15% flagged): opening turn bundles multiple concerns
- **Calm persona contamination** (34% of calm conversations): mild frustration markers leak across the boundary
- **Depth-first and info-drip correlated**: same root cause — model front-loads goals and facts
- **Expert personas hardest** (mean Layer 2 = 4.38 vs novice = 4.63)

**First results on failure conversations (Layer 1 only):**
- `DROPOUT_NOT_PLAUSIBLE` flagged in 41% — most common new failure mode: customer exits but the final message is too generic to clearly signal the assigned dropout trigger
- `CERTAINTY_EXCEEDS_HEDGING` persists at a similar rate — over-confidence is not specific to goal-completion scenarios

---

## Pass Rates (Success Conversations)

| Tier | Threshold | Pass Rate |
|---|---|---|
| Layer 1 — clean (zero warnings) | 0 warnings | 57.5% (46/80) |
| Layer 1 — not outlier | < 3 warnings | 96.3% (77/80) |
| Layer 2 — high quality | ≥ 4.0 score | 83.3% (60/72 scoreable) |
| Layer 2 — top tier | ≥ 4.5 score | 58.3% (42/72 scoreable) |

*Pass rates for failure, type_a, and type_b to be filled once full generation and Layer 2 evaluation runs complete.*
