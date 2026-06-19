# UserLM Data Pipeline Overhaul — Execution Plan

**Audience:** a Claude coding agent working in this repo.
**Scope:** fix the synthetic-data generation pipeline so the training corpus has real content diversity, a realistic length distribution, and no spurious correlations. Data generation is **Claude-API-based (no GPU)**. This plan does **not** cover RL/Mistral training — that is downstream and out of scope here.

**Repo paths (relative to repo root):**
- `synthetic_data_generation/generate_conversations.py` — success + failure modes
- `synthetic_data_generation/generate_probe_conversations.py` — type_a (inadvertent) + type_b (adversarial)
- `synthetic_data_generation/split_dataset.py` — combines + splits (conversation-level, leakage-free; **keep working**)
- `synthetic_data_generation/run_all.sh` — orchestration
- `evaluate_predictions/eval_conversation_predictions.py` — eval (must keep consuming the same JSONL schema)

---

## 0. Why (evidence from current data — do not re-litigate, just fix)

Measured on the existing JSONL files:

| File | entries | distinct merchants | distinct personas | user-msg word len (mean / median / p90) |
|---|---|---|---|---|
| userlm_success | 706 | **8** | 10 | 9.6 / 9 / 19 |
| userlm_type_a | 345 | **8** | 5 | 11.3 / 10 / 23 |
| userlm_type_b | 295 | **8** | 4 | 13.4 / 13 / 25 |
| userlm_failure | 365 | **8** | 5 | 12.1 / 11 / 21 |

Root causes to eliminate:
1. **Content is hardcoded.** `merchant_name`, `amount`, `transaction_date` are baked into the 8 `SCENARIOS` templates → the entire corpus reuses **8 merchants / 8 amounts / one August date window**.
2. **Spurious 1:1 correlations:** `failure` ties persona→one failure mode (`PERSONA_FAILURE_MAP`); `type_a` ties scenario→one prior belief (`PRIOR_BELIEFS`); `type_b` ties scenario→one planted error (`PLANTED_ERRORS`). Model overfits surface features → poor validation on a/b/failure.
3. **Length is clamped globally** ("1-2 sentences. Rarely more", "terse 3-6 words") and **no `verbose` persona is active**, so the long tail is missing → eval shows ~13% length deflation (`token_ratio ≈ 0.87`).
4. **Class imbalance** (success ≈ 2× each other type).
5. **`run_all.sh` has stale `--limit` values** (failure=1, type_a=5, type_b=18) — not `all`.
6. **Validation reuses the same 8 merchants** → cannot measure content generalization.

---

## Hard constraints (do NOT break)

- **Preserve the training JSONL schema exactly:** each row is `{"input": str, "output": str, "_meta": {...}}`. `output` ends with `<eos>`; final user turn = bare `<eos>`. `split_dataset.py` detects type from `_meta.probing_type` (`inadvertent`/`adversarial`) and `_meta.goal_completed`. **Every existing `_meta` field must still be written**, including `probing_type` for probe data. Verify before and after.
- **Keep `parse_conversation`, `to_training_format`, and the CLI backward compatible.** Add new args with safe defaults; do not rename existing ones.
- **Conversation-level split stays leakage-free.** Do not change `split_dataset.py`'s split logic.
- **Widen `generation_idx` ranges** so larger volumes don't collide (the split has a collision guard that will hard-fail). New offsets:
  - success: `0 – 99,999`
  - failure: `100,000 – 199,999`
  - type_a: `200,000 – 299,999`
  - type_b: `300,000 – 399,999`
- **Cost discipline:** always run `--provider dry_run` first and inspect prompts; then a small sample (`--limit 5`), eyeball the human-readable dumps in `conversation_dumps_*`; only then full run.

---

## Phase 1 — Decouple content from structure (highest leverage)

**Goal:** content (merchant, amount, date, card, channel) is sampled independently per conversation, not fixed per scenario.

### Task 1.1 — Create `synthetic_data_generation/content_pools.py`
A new module with sampling helpers. Suggested contents:

```python
import random

MERCHANTS = [  # >= 50 entries, mix of categories
    "Amazon", "Walmart", "Target", "Best Buy", "Apple Store", "Netflix",
    "Spotify", "Uber", "Uber Eats", "DoorDash", "Lyft", "Shell", "Chevron",
    "Starbucks", "Costco", "Home Depot", "Steam", "PlayStation Store",
    "Adobe", "Microsoft", "Google Play", "Airbnb", "Expedia", "Delta Air Lines",
    "CVS Pharmacy", "Walgreens", "Nike", "Etsy", "eBay", "PayPal",
    "Cash App", "Venmo", "Zelle transfer", "Wayfair", "Instacart", "GrubHub",
    "AT&T", "Verizon", "Comcast", "Planet Fitness", "Peloton", "Ticketmaster",
    "an unfamiliar online store", "a gas station I don't recognize",
    "some subscription I don't remember", "a foreign merchant", "Temu",
    "Shein", "AliExpress", "a hotel in another city",
]
# Held-out merchants reserved ONLY for the generalization test set (Phase 5)
MERCHANTS_HELDOUT = [
    "Chipotle", "REI", "Sephora", "Square checkout", "Roblox",
    "a parking garage", "a medical clinic", "an unknown ATM withdrawal",
]

CARD_TYPES = ["Visa", "Mastercard", "Amex", "debit card", "credit card"]
CHANNELS   = ["online", "in-store", "recurring subscription", "ATM", "phone order"]

def sample_amount(rng: random.Random) -> str:
    # log-uniform $3–$3000, mostly small; ~15% round numbers
    import math
    lo, hi = math.log(3), math.log(3000)
    val = math.exp(rng.uniform(lo, hi))
    if rng.random() < 0.15:
        val = round(val / 10) * 10  # round number
    return f"${val:,.2f}"

def sample_date(rng: random.Random, ref_date=None) -> str:
    # spread over a ~180-day rolling window, varied month names
    import datetime
    ref = ref_date or datetime.date.today()
    d = ref - datetime.timedelta(days=rng.randint(1, 180))
    return d.strftime("%B %-d")  # e.g. "March 4"

def sample_content(rng: random.Random, heldout: bool = False) -> dict:
    pool = MERCHANTS_HELDOUT if heldout else MERCHANTS
    return {
        "merchant_name": rng.choice(pool),
        "amount": sample_amount(rng),
        "transaction_date": sample_date(rng),
        "card_type": rng.choice(CARD_TYPES),
        "channel": rng.choice(CHANNELS),
    }
```

### Task 1.2 — Inject sampled content in both generators
- In `generate_conversations.py` and `generate_probe_conversations.py`, the `Scenario` dataclass keeps the **structural** fields (`certainty`, `information_completeness`, `prior_contact`, `expected_resolution`) but **content fields become per-conversation samples**, not template constants.
- Replace the hardcoded `SCENARIOS` content with a structural-only template list (8 structural templates is fine), then call `sample_content(rng)` for each generated conversation and merge into the scenario used by `_scenario_description`, `generate_intent_summary`, and `_meta.scenario`.
- Use a **per-conversation seeded RNG** derived from `(base_seed, global_idx)` for reproducibility.

**Acceptance 1:** regenerated training data has **≥ 40 distinct merchants**, **≥ 100 distinct amounts**, and a **date span ≥ 120 days** in *each* of the 4 types. `_meta.scenario` still present with the new content.

---

## Phase 2 — Break the spurious correlations

### Task 2.1 — Failure mode (in `generate_conversations.py`)
- Delete the deterministic `PERSONA_FAILURE_MAP` lookup. Sample `failure_mode` **independently** per conversation via `rng.choice(list(FAILURE_MODE_INSTRUCTIONS))`.
- **Enable all 8 failure modes** (currently only 5 are reachable; `escalation_exit`, `wrong_channel`, `distraction` are unused).
- Keep writing `_meta.failure_mode` and `_meta.goal_completed = False`.

### Task 2.2 — Type A prior belief (in `generate_probe_conversations.py`)
- Convert `PRIOR_BELIEFS` from a `{scenario_idx: belief}` dict into a **flat pool** of plausible misbeliefs. Sample independently per conversation.
- Optionally tag each belief with the scenario structural-types it's *compatible* with, and sample only among compatible ones — but **never** a 1:1 mapping. Record chosen belief in `_meta`.

### Task 2.3 — Type B planted error (in `generate_probe_conversations.py`)
- Convert `PLANTED_ERRORS` from `{scenario_idx: error}` into a **flat pool** of `PlantedError` objects. Sample independently per conversation; also randomize `timing`.
- Keep writing `_meta.probing_type = "adversarial"` and record the chosen error mode in `_meta`.

**Acceptance 2:** in regenerated data — all 8 failure modes present; **each persona co-occurs with ≥ 3 different failure modes**; **each structural scenario co-occurs with ≥ 3 different planted-error modes** and **≥ 3 different prior beliefs**. (Add these checks to the verify script in Phase 6.)

---

## Phase 3 — Restore the length distribution

### Task 3.1 — Re-enable long personas
- Uncomment / add `verbose` personas in both `PERSONAS` lists so `communication_style == "verbose"` is actually generated.

### Task 3.2 — Per-conversation target length
- Add a `sample_target_length(rng, persona)` helper returning a soft word-count band, drawn from a long-tailed mixture, modulated by persona:
  - terse → mostly 3–12 words
  - direct → 8–25
  - indirect/verbose → 20–70, with a tail to ~90
- Inject as **soft guidance** into the user/system prompt (e.g., "Your messages in this chat tend to run about N words; vary naturally around that.").
- **Soften the global brevity clamp:** remove or down-weight "Customer messages are typically 1-2 sentences. Rarely more." so length is **persona-driven**, not globally capped.

### Task 3.3 — Filter degenerate turns
- Drop or cap 1-word user turns (allow but keep < 5% of turns).

**Acceptance 3:** regenerated combined training data has **median user-msg ≥ 12 words**, **p90 ≥ 35 words**, **≥ 10% of user msgs > 30 words**, and **< 5% one-word** user msgs.

---

## Phase 4 — Rebalance + few-shot integrity

### Task 4.1 — Balance the 4 types
- Make per-type conversation counts configurable and roughly equal (default target: ~500 conversations each, ±15%). Expose as a `--target-conversations` arg or constant.

### Task 4.2 — Verify the few-shot source is REAL human data
- `--data-path` feeds few-shot examples. Confirm `userlm_data.jsonl` is **real human** conversations, not prior synthetic output. If it is synthetic, switch the few-shot source to a dedicated real-seed file (e.g., `real_seed.jsonl`) and document it. Self-distillation here amplifies mode collapse.
- Cap few-shot at 3 examples (already the case) and ensure they are sampled from the **real** pool only.

**Acceptance 4:** per-type counts within ±15%; few-shot source documented and confirmed real (or flagged clearly in the PR description if it cannot be confirmed).

---

## Phase 5 — Held-out generalization test set

### Task 5.1 — Generate a disjoint test set
- Add a generation mode/flag that uses `sample_content(rng, heldout=True)` (the `MERCHANTS_HELDOUT` pool + amounts/dates outside the training window) so test content is **disjoint** from training.
- Output `test_heldout.jsonl` with a distinct `generation_idx` range (`400,000+`).

### Task 5.2 — Wire into eval
- Ensure `evaluate_predictions/eval_conversation_predictions.py` can score this file unchanged (same schema). Add it as a reported slice so content-generalization is measured separately from in-distribution val.

**Acceptance 5:** `test_heldout.jsonl` shares **0 merchants** with the training pool; eval runs on it without code changes.

---

## Phase 6 — Verification script (gate before any full run)

### Task 6.1 — Create `synthetic_data_generation/verify_dataset.py`
A standalone script that loads any combination of the JSONL files and **asserts** the acceptance criteria above, printing a pass/fail table. It must check:
- distinct merchants / amounts / date-span per type (Acceptance 1)
- failure-mode coverage + persona↔failure-mode and scenario↔error decorrelation (Acceptance 2)
- length distribution stats (Acceptance 3)
- per-type balance (Acceptance 4)
- held-out disjointness (Acceptance 5)
- schema integrity: every row has `input`/`output`/`_meta`; `output` ends with `<eos>`; probe rows carry `probing_type`; no `generation_idx` collisions across files.

The script exits non-zero on any failure so it can gate CI / the run script.

---

## Phase 7 — Fix orchestration

### Task 7.1 — `run_all.sh`
- Set **all** generation steps to `--limit all` (or to `--target-conversations`).
- Add the held-out test generation step (Phase 5) and a final `python verify_dataset.py` gate before declaring success.
- Update the generation_idx offset comments to the new ranges.

---

## Execution order & dependencies

1. Phase 1 (content pools) — foundational, do first.
2. Phase 2 + Phase 3 — independent of each other, both depend on Phase 1.
3. Phase 4 — depends on 1–3.
4. Phase 6 (verify script) — write **early**, run after each phase on a small sample.
5. Phase 5 + Phase 7 — last.

**Per-phase loop:** edit → `--provider dry_run` and read 3–5 prompts → `--limit 5` real run → inspect `conversation_dumps_*` → `verify_dataset.py` on the sample → proceed.

## Definition of done
- All acceptance criteria pass in `verify_dataset.py` on a full regeneration.
- `split_dataset.py` and `eval_conversation_predictions.py` run unchanged on the new data.
- A short `PR_NOTES.md` summarizing: new diversity/length numbers vs. the table in §0, few-shot source confirmation, and any decision points left for the human.

## Decision points to surface to the human (do not guess)
- Final per-type conversation count / total API budget.
- Whether to fully remove vs. merely soften the brevity rule (affects realism vs. length).
- Confirmation that `userlm_data.jsonl` is real human data (Task 4.2).
