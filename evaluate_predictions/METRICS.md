# UserLM Prediction Evaluation — Metrics Reference

---

## eval_predictions.py — turn-level

| Metric | What it measures | How it's computed |
|---|---|---|
| BLEU / ROUGE-L / METEOR | N-gram overlap vs reference turn | Pre-computed during inference, read from `model_data["metrics"]` dict. Not recomputed here. |
| BERTScore F1 | Semantic similarity via embeddings | Pre-computed during inference, read from same metrics dict. |
| Exact match / EOS recall | Correct end-of-conversation prediction | `expected.strip() == "<eos>"` identifies EOS turns; correct prediction = `predicted == ""` (empty string). |
| EOS false positive rate | Model ends early when it shouldn't | On non-EOS turns, checks if `predicted == ""`. |
| Style compliance | Predicted turn matches persona style | Rule-based. Checks word count thresholds, jargon presence, and formality signals against persona fields. Suppressed for type_b turns. |
| Hedge / promise / certainty rate | Lexical tone signals | Rule-based phrase list matching (e.g. "I think", "I'll make sure") counted over token-normalized text. |
| Type-token ratio | Lexical diversity | `len(unique_words) / len(all_words)` on predicted and reference separately; delta taken. |
| Entity precision/recall | Named entities carried over correctly | Rule-based extraction of amounts (`$\d+`), dates, merchant names; set intersection vs reference. |
| Role confusion | Model wrote from agent's perspective | Regex for "Agent:" prefix + phrase list (e.g. "I can help you with", "our records show"). |
| is_pushback_turn (type_b) | Turn contains error-correction language | Phrase list match against `PUSHBACK_PHRASES` (20 phrases like "that's not right", "i was told"). Only evaluated for type_b. |
| prior_belief_expressed (type_a) | Turn expresses the wrong prior belief | Keyword match against `PRIOR_BELIEF_KEYWORDS[scenario_idx]` — per-scenario keyword sets. Only evaluated for type_a. |

---

## eval_conversation_predictions.py — conversation-level

### Tier 1 — rule-based

| Metric | What it measures | How it's computed |
|---|---|---|
| Word length mean/std/p10/p90 | Turn length distribution vs reference | `word_count()` on each turn, then `safe_mean()`, `safe_std()`, `percentile()` over all turns. Delta = pred − ref. |
| Hedge / certainty / promise rate | Tone trajectory across full conversation | Phrase list counts divided by turn count, averaged across all turns. Same lists as turn-level eval. |
| Info density / facts in turn 1 | Customer front-loading information | Counts how many fact categories (amount, date, merchant, card) appear in turn 1 specifically. |
| Breadth-first flag | Customer raises multiple topics at once | Checks if turn 1 mentions ≥2 distinct fact categories — signals unnatural information volunteering. |
| Pushback count + first index (type_b) | Error catch rate and timing | `PUSHBACK_PHRASES` match across all turns; counts hits and records index of first hit. |
| Prior belief rate (type_a) | Fraction of turns expressing wrong belief | `PRIOR_BELIEF_KEYWORDS[scenario_idx]` match; hits divided by total turns. |
| Dropout turn index (failure) | Which turn the customer exits at | `len(predicted_turns) - 1` — the last turn is definitionally the exit point. |
| flag_dropout_implausible (failure) | Exit lacks authentic dropout language | Checks last predicted turn against `DROPOUT_EXIT_PHRASES` (17 phrases). Flagged if none match. |
| flag_missed_pushback (type_b) | Reference catches error but prediction doesn't | `ref_pushback_count > 0 and pred_pushback_count == 0`. |
| flag_wrong_belief_missing (type_a) | Reference expresses prior belief, prediction doesn't | `ref_prior_belief_rate > 0.0 and pred_prior_belief_rate == 0.0`. |
| Turn count fidelity | Predicted conversation is structurally longer or shorter than reference | `turn_count_delta = len(predicted_turns) − len(reference_turns)`. Raises `flag_turn_count_mismatch` when `abs(delta) ≥ 2` — single-turn variance is normal; 2+ signals a structural difference. |
| Escalation trajectory *(escalating personas only)* | Frustration builds monotonically across turns rather than being flat or decreasing | Counts `MILD_FRUSTRATION_MARKERS + STRONG_FRUSTRATION_MARKERS` hits per turn, then computes Spearman correlation between turn position and hit count via `scipy.stats.spearmanr`. Returns `None` for fewer than 3 turns. Raises `flag_escalation_flat` when `pred_escalation_corr < 0.2` while `ref_escalation_corr ≥ 0.2`. |
| Cross-turn self-contradiction *(predicted turns only)* | Predicted customer contradicts a factual claim made in an earlier turn | Tracks first-stated amount (regex on `$X` patterns), date (month + day pattern), and merchant name (hardcoded set: Amazon, Netflix, Shell, TechHub, Uber, Adobe, Apple) across turns. Returns `True` on first conflict. Not run on reference turns — self-correction in reference is intentional. Raises `flag_self_contradiction` if `True`. |

### Tier 2 — LLM-as-judge (1–5)

| Type | Dimensions scored | How it's computed |
|---|---|---|
| success | Depth-First, Uncertainty, Info Drip, Pragmatic, Persona | Single Claude API call with full predicted + reference turns. JSON parsed for scores; weighted mean for overall. |
| failure | Dropout Authenticity, Depth-First, Uncertainty, Pragmatic, Persona | Same structure; judge prompt includes `failure_mode` so it knows what exit trigger to expect. |
| type_a | Prior Belief Persistence, Info Incompleteness, Depth-First, Uncertainty, Persona | Judge prompt includes the specific wrong prior belief text and `information_completeness` value. |
| type_b | Error Detection, Pushback Calibration, Uncertainty, Pragmatic, Persona | Judge prompt names the planted `agent_failure_mode` explicitly so judge knows what error to look for. |
