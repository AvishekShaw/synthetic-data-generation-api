"""
rl_train.py
-----------
GRPO-based RL training for the UserLM (Gemma-3-4B-IT + LoRA).

Pipeline
────────
  1. Load base model (Gemma-3-4B-IT)
  2. Load SFT LoRA adapter (rank-8 from SFT checkpoint)
  3. Merge adapter into base weights  →  this merged model is also the
     frozen reference for GRPO's KL penalty (no extra copy needed)
  4. Attach fresh rank-16 LoRA adapters for RL exploration
  5. Load train JSONL; expose `input` field as `prompt` column
  6. Build composite reward:  judge_reward (0.7)  +  rule_reward (0.3)
  7. Run GRPOTrainer
  8. Save final LoRA adapter

Launch via:
    accelerate launch --config_file accelerate_config.yaml rl_train.py [args]
    (see launch_rl.sh for a ready-made command)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer

from rl_config import RLTrainingConfig
from reward_functions import make_judge_reward_fn, make_rule_based_reward_fn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_model_for_rl(cfg: RLTrainingConfig) -> torch.nn.Module:
    """
    Three-step setup:
      base  →  merge(SFT adapter)  →  attach fresh RL LoRA adapters

    After merging, the base weights encode all SFT learning.
    Fresh LoRA adapters initialised at zero give RL the rank-16 capacity
    it needs without destroying the SFT starting point.

    The merged weights also serve as the implicit GRPO reference model:
    GRPOTrainer runs reference forward passes with LoRA disabled, so no
    second copy of the model is needed in memory.
    """
    logger.info(f"Loading base model from: {cfg.base_model_path}")
    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        # device_map=None  →  let accelerate / FSDP handle placement
        device_map=None,
    )

    logger.info(f"Loading SFT adapter from: {cfg.sft_adapter_path}")
    model = PeftModel.from_pretrained(base, cfg.sft_adapter_path)

    logger.info("Merging SFT adapter into base weights …")
    model = model.merge_and_unload()

    logger.info(f"Attaching fresh LoRA adapters (rank={cfg.lora_rank}) for RL …")
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        # init_lora_weights="gaussian" is an option for more initial diversity,
        # but the default (zeros for B, random for A) is fine here.
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    return model


# ══════════════════════════════════════════════════════════════════════════════
# Tokenizer
# ══════════════════════════════════════════════════════════════════════════════

def load_tokenizer(cfg: RLTrainingConfig) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Left-padding is required for batched generation in decoder-only models.
    tokenizer.padding_side = "left"

    return tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

def load_rl_dataset(data_path: str) -> Dataset:
    """
    Load train JSONL and expose:
      - "prompt"         : the `input` field (FIRST_TURN or COMPLETION template)
                           — this is what GRPOTrainer feeds to the model
      - "reference"      : the gold `output` — not used for training but useful
                           for side-by-side logging / debugging
      - "generation_idx" : conversation ID from _meta — useful for grouping logs

    GRPOTrainer only strictly needs the "prompt" column; extras are passed
    through to reward functions via **kwargs.
    """
    records = []
    with open(data_path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed line {line_no}: {e}")
                continue

            records.append({
                "prompt":         row["input"],
                "reference":      row.get("output", ""),
                "generation_idx": row.get("_meta", {}).get("generation_idx", -1),
            })

    dataset = Dataset.from_list(records)
    logger.info(f"Loaded {len(dataset):,} RL training prompts from {data_path}")
    return dataset


# ══════════════════════════════════════════════════════════════════════════════
# CLI argument parser
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GRPO RL training for UserLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Allow all key paths to be overridden from the command line so you can
    # run experiments without editing the config file.
    p.add_argument("--base_model_path",   type=str, default=None)
    p.add_argument("--sft_adapter_path",  type=str, default=None)
    p.add_argument("--train_data_path",   type=str, default=None)
    p.add_argument("--output_dir",        type=str, default=None)
    p.add_argument("--lora_rank",         type=int, default=None)
    p.add_argument("--num_generations",   type=int, default=None,
                   help="G: completions generated per prompt (GRPO group size)")
    p.add_argument("--beta",              type=float, default=None,
                   help="KL penalty coefficient")
    p.add_argument("--learning_rate",     type=float, default=None)
    p.add_argument("--judge_api_url",     type=str, default=None)
    p.add_argument("--judge_model",       type=str, default=None)
    p.add_argument("--judge_api_key",     type=str, default=None)
    return p.parse_args()


def apply_cli_overrides(cfg: RLTrainingConfig, args: argparse.Namespace) -> None:
    """Overwrite config fields with any non-None CLI arguments."""
    for field_name, value in vars(args).items():
        if value is not None and hasattr(cfg, field_name):
            setattr(cfg, field_name, value)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    cfg = RLTrainingConfig()
    apply_cli_overrides(cfg, args)

    # ── Sanity checks ────────────────────────────────────────────────────────
    if "placeholder" in cfg.judge_api_url.lower():
        logger.warning(
            "judge_api_url still contains 'placeholder' — "
            "judge rewards will return 0.0. "
            "Set --judge_api_url / --judge_model / --judge_api_key before a real run."
        )

    # ── Build model and tokenizer ────────────────────────────────────────────
    tokenizer = load_tokenizer(cfg)
    model = load_model_for_rl(cfg)

    # ── Build dataset ────────────────────────────────────────────────────────
    dataset = load_rl_dataset(cfg.train_data_path)

    # ── Build reward functions ───────────────────────────────────────────────
    #
    # Passing two functions to GRPOTrainer with reward_weights=[0.7, 0.3] means:
    #   total_reward = 0.7 * judge_reward + 0.3 * rule_reward
    #
    # TRL logs each component separately in wandb:
    #   rewards/judge_reward   and   rewards/rule_reward
    # This lets you see which dimension is the binding constraint.
    #
    judge_reward_fn = make_judge_reward_fn(
        api_url=cfg.judge_api_url,
        model=cfg.judge_model,
        api_key=cfg.judge_api_key,
        timeout=cfg.judge_api_timeout,
        max_workers=cfg.judge_max_workers,
    )
    rule_reward_fn = make_rule_based_reward_fn(
        tokenizer=tokenizer,
        min_tokens=cfg.min_completion_tokens,
        max_tokens=cfg.max_completion_tokens,
    )

    # ── GRPO trainer ─────────────────────────────────────────────────────────
    grpo_config = cfg.get_grpo_config()

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[judge_reward_fn, rule_reward_fn],
        reward_weights=[cfg.judge_reward_weight, cfg.rule_reward_weight],
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting GRPO training …")
    trainer.train()

    # ── Save ─────────────────────────────────────────────────────────────────
    save_path = os.path.join(cfg.output_dir, "final_adapter")
    logger.info(f"Saving final RL LoRA adapter to {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    logger.info("RL training complete.")


if __name__ == "__main__":
    main()
