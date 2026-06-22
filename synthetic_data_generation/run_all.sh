#!/bin/bash
set -e

# ─────────────────────────────────────────────────────────────────────────────
# UserLM full generation pipeline
#
# generation_idx ranges:
#   success        →   0 –  99,999
#   failure        → 100,000 – 199,999
#   type_a         → 200,000 – 299,999
#   type_b         → 300,000 – 399,999
#   heldout success→ 400,000 – 409,999
#   heldout failure→ 410,000 – 419,999
#   heldout type_a → 420,000 – 429,999
#   heldout type_b → 430,000 – 439,999
#
# Cost discipline:
#   1. Run with --provider dry_run first and review prompts.
#   2. Run with --target-conversations 5 and eyeball conversation_dumps_*.
#   3. Run verify_dataset.py on the sample.
#   4. Only then set TARGET_CONVERSATIONS to the full target and re-run.
# ─────────────────────────────────────────────────────────────────────────────

TARGET_CONVERSATIONS="${TARGET_CONVERSATIONS:-500}"   # per type; override via env
HELDOUT_CONVERSATIONS="${HELDOUT_CONVERSATIONS:-100}" # per type for test set

SUCCESS_OUT="userlm_success.jsonl"
FAILURE_OUT="userlm_failure.jsonl"
TYPE_A_OUT="userlm_type_a.jsonl"
TYPE_B_OUT="userlm_type_b.jsonl"

HELDOUT_SUCCESS_OUT="heldout_success.jsonl"
HELDOUT_FAILURE_OUT="heldout_failure.jsonl"
HELDOUT_TYPE_A_OUT="heldout_type_a.jsonl"
HELDOUT_TYPE_B_OUT="heldout_type_b.jsonl"
HELDOUT_OUT="test_heldout.jsonl"

DATA_PATH="userlm_data.jsonl"

# ── 1. Success conversations ──────────────────────────────────────────────────
echo "=== [1/8] Generating SUCCESS conversations ==="
python generate_conversations.py \
  --mode success \
  --target-conversations "$TARGET_CONVERSATIONS" \
  --provider anthropic \
  --data-path "$DATA_PATH" \
  --output "$SUCCESS_OUT"

# ── 2. Failure / dropout conversations ───────────────────────────────────────
echo "=== [2/8] Generating FAILURE conversations ==="
python generate_conversations.py \
  --mode failure \
  --target-conversations "$TARGET_CONVERSATIONS" \
  --provider anthropic \
  --data-path "$DATA_PATH" \
  --output "$FAILURE_OUT"

# ── 3. Type-A inadvertent probing ─────────────────────────────────────────────
echo "=== [3/8] Generating TYPE-A probe conversations ==="
python generate_probe_conversations.py \
  --mode type_a \
  --target-conversations "$TARGET_CONVERSATIONS" \
  --provider anthropic \
  --output "$TYPE_A_OUT"

# ── 4. Type-B adversarial probing ─────────────────────────────────────────────
echo "=== [4/8] Generating TYPE-B probe conversations ==="
python generate_probe_conversations.py \
  --mode type_b \
  --target-conversations "$TARGET_CONVERSATIONS" \
  --provider anthropic \
  --output "$TYPE_B_OUT"

# ── 5. Held-out test set ──────────────────────────────────────────────────────
echo "=== [5/8] Generating HELD-OUT success conversations ==="
python generate_conversations.py \
  --mode success \
  --heldout \
  --target-conversations "$HELDOUT_CONVERSATIONS" \
  --provider anthropic \
  --data-path "$DATA_PATH" \
  --output "$HELDOUT_SUCCESS_OUT"

echo "=== [6/8] Generating HELD-OUT failure conversations ==="
python generate_conversations.py \
  --mode failure \
  --heldout \
  --target-conversations "$HELDOUT_CONVERSATIONS" \
  --provider anthropic \
  --data-path "$DATA_PATH" \
  --output "$HELDOUT_FAILURE_OUT"

echo "=== [7/8] Generating HELD-OUT type-a conversations ==="
python generate_probe_conversations.py \
  --mode type_a \
  --heldout \
  --target-conversations "$HELDOUT_CONVERSATIONS" \
  --provider anthropic \
  --output "$HELDOUT_TYPE_A_OUT"

echo "=== [7/8] Generating HELD-OUT type-b conversations ==="
python generate_probe_conversations.py \
  --mode type_b \
  --heldout \
  --target-conversations "$HELDOUT_CONVERSATIONS" \
  --provider anthropic \
  --output "$HELDOUT_TYPE_B_OUT"

echo "=== Combining held-out files → $HELDOUT_OUT ==="
cat "$HELDOUT_SUCCESS_OUT" "$HELDOUT_FAILURE_OUT" "$HELDOUT_TYPE_A_OUT" "$HELDOUT_TYPE_B_OUT" \
  > "$HELDOUT_OUT"

# ── 8. Verify before splitting ───────────────────────────────────────────────
echo "=== [8/8] Running verify_dataset.py ==="
python verify_dataset.py \
  --success "$SUCCESS_OUT" \
  --failure "$FAILURE_OUT" \
  --type-a  "$TYPE_A_OUT" \
  --type-b  "$TYPE_B_OUT" \
  --heldout "$HELDOUT_OUT" \
  --target-conversations "$TARGET_CONVERSATIONS"

# ── 9. Combine + split ────────────────────────────────────────────────────────
echo "=== [9/9] Splitting training dataset 80:20 conversation-wise ==="
python split_dataset.py \
  --files "$SUCCESS_OUT" "$FAILURE_OUT" "$TYPE_A_OUT" "$TYPE_B_OUT" \
  --ratio 0.8 \
  --seed 42 \
  --train-out train_combined.jsonl \
  --val-out val_combined.jsonl

echo ""
echo "Done. Files written:"
echo "  train_combined.jsonl"
echo "  val_combined.jsonl"
echo "  $HELDOUT_OUT  (disjoint test set)"
