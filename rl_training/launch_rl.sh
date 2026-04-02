#!/usr/bin/env bash
# launch_rl.sh
# ------------
# Launch GRPO RL training across 8× L40s GPUs using HuggingFace Accelerate.
#
# Prerequisites:
#   pip install trl>=0.12 peft transformers accelerate wandb
#
# Usage:
#   bash launch_rl.sh                          # uses defaults from rl_config.py
#   bash launch_rl.sh --lora_rank 32           # override any field
#   DRY_RUN=1 bash launch_rl.sh               # just print the command

set -euo pipefail

# ── Configurable paths (override here or via env vars) ──────────────────────
BASE_MODEL="${BASE_MODEL:-google/gemma-3-4b-it}"
SFT_ADAPTER="${SFT_ADAPTER:-./sft_checkpoint/adapter}"
TRAIN_DATA="${TRAIN_DATA:-./data/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./rl_checkpoint}"

# ── Internal judge API (set these before a real run) ────────────────────────
JUDGE_API_URL="${JUDGE_API_URL:-http://YOUR_INTERNAL_API/v1/chat/completions}"
JUDGE_MODEL="${JUDGE_MODEL:-YOUR_INTERNAL_MODEL_NAME}"
JUDGE_API_KEY="${JUDGE_API_KEY:-YOUR_API_KEY}"

# ── Extra args forwarded to rl_train.py ─────────────────────────────────────
EXTRA_ARGS="${@}"   # e.g. --lora_rank 32 --beta 0.005

# ── W&B run name (optional, remove if not using wandb) ──────────────────────
export WANDB_PROJECT="userlm-rl"
export WANDB_RUN_NAME="grpo-gemma3-4b-$(date +%Y%m%d-%H%M)"

# ── Build the command ────────────────────────────────────────────────────────
CMD=(
    accelerate launch
        --num_processes 8
        --mixed_precision bf16
        # For a 4B model on 8× L40s, DDP is sufficient (no FSDP needed).
        # If you later move to 27B, switch to:
        #   --use_fsdp
        #   --fsdp_sharding_strategy FULL_SHARD
        #   --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP
        --dynamo_backend no
    rl_train.py
        --base_model_path  "$BASE_MODEL"
        --sft_adapter_path "$SFT_ADAPTER"
        --train_data_path  "$TRAIN_DATA"
        --output_dir       "$OUTPUT_DIR"
        --judge_api_url    "$JUDGE_API_URL"
        --judge_model      "$JUDGE_MODEL"
        --judge_api_key    "$JUDGE_API_KEY"
        $EXTRA_ARGS
)

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  UserLM GRPO RL Training"
echo "  Model   : $BASE_MODEL"
echo "  Adapter : $SFT_ADAPTER"
echo "  Data    : $TRAIN_DATA"
echo "  Output  : $OUTPUT_DIR"
echo "  W&B run : $WANDB_RUN_NAME"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[DRY RUN] Would execute:"
    echo "  ${CMD[*]}"
    exit 0
fi

exec "${CMD[@]}"
