#!/bin/bash
set -e

# ─────────────────────────────────────────────────────────────────────────────
# UserLM full generation pipeline
#
# Pass 1 — success (bootstrap, no prior data)
# Pass 2 — failure dropout (uses pass-1 output as few-shot)
# Pass 3 — type_a probing
# Pass 4 — type_b adversarial probing
# Final  — combine all 4 JSONL files and split 80:20 conversation-wise
# ─────────────────────────────────────────────────────────────────────────────

SUCCESS_OUT="userlm_success.jsonl"
FAILURE_OUT="userlm_failure.jsonl"
TYPE_A_OUT="userlm_type_a.jsonl"
TYPE_B_OUT="userlm_type_b.jsonl"

DATA_PATH="userlm_data.jsonl"

# ── 1. Success conversations ──────────────────────────────────────────────────
echo "=== [1/4] Generating SUCCESS conversations ==="
python generate_conversations.py \
  --mode success \
  --limit all \
  --provider anthropic \
  --data-path "$DATA_PATH" \
  --output "$SUCCESS_OUT"

# ── 2. Failure / dropout conversations ───────────────────────────────────────
echo "=== [2/4] Generating FAILURE conversations ==="
python generate_conversations.py \
  --mode failure \
  --limit 40 \
  --provider anthropic \
  --data-path "$DATA_PATH" \
  --output "$FAILURE_OUT"

# ── 3. Type-A inadvertent probing ─────────────────────────────────────────────
echo "=== [3/4] Generating TYPE-A probe conversations ==="
python generate_probe_conversations.py \
  --mode type_a \
  --limit all \
  --provider anthropic \
  --output "$TYPE_A_OUT"

# ── 4. Type-B adversarial probing ─────────────────────────────────────────────
echo "=== [4/4] Generating TYPE-B probe conversations ==="
python generate_probe_conversations.py \
  --mode type_b \
  --limit all \
  --provider anthropic \
  --output "$TYPE_B_OUT"

# ── 5. Combine + split ────────────────────────────────────────────────────────
echo "=== [5/5] Splitting dataset 80:20 conversation-wise ==="
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
