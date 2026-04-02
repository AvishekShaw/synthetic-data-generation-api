"""
rl_config.py
------------
Central configuration for the GRPO-based RL training run.

Two objects live here:
  - RLTrainingConfig   : everything specific to this project (paths, LoRA dims,
                         reward weights, API credentials)
  - get_grpo_config()  : returns a trl.GRPOConfig populated from the above

Edit RLTrainingConfig fields directly or override via CLI args in rl_train.py.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
from trl import GRPOConfig


@dataclass
class RLTrainingConfig:
    # ------------------------------------------------------------------ #
    # Model paths                                                          #
    # ------------------------------------------------------------------ #
    base_model_path: str = "google/gemma-3-4b-it"
    # Directory containing adapter_config.json + adapter_model.safetensors
    # produced by your SFT run (rank-8 checkpoint).
    sft_adapter_path: str = "./sft_checkpoint/adapter"
    output_dir: str = "./rl_checkpoint"

    # ------------------------------------------------------------------ #
    # LoRA config for the RL phase (fresh adapters on the merged base)     #
    # Higher rank than SFT (4/8) to give RL more room to explore.         #
    # ------------------------------------------------------------------ #
    lora_rank: int = 16
    lora_alpha: int = 32          # typically 2× rank
    lora_dropout: float = 0.05
    # Target all projection matrices — same targets you used for SFT.
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # ------------------------------------------------------------------ #
    # Data                                                                 #
    # ------------------------------------------------------------------ #
    train_data_path: str = "./data/train.jsonl"

    # ------------------------------------------------------------------ #
    # Reward weights (must sum to 1.0)                                     #
    # judge_reward  : LLM-as-judge composite score  (expensive, high signal)
    # rule_reward   : fast rule-based checks         (cheap, hard floor)
    # ------------------------------------------------------------------ #
    judge_reward_weight: float = 0.7
    rule_reward_weight: float = 0.3

    # ------------------------------------------------------------------ #
    # Internal judge API  (replace placeholders before running)           #
    # ------------------------------------------------------------------ #
    judge_api_url: str = "http://YOUR_INTERNAL_API/v1/chat/completions"
    judge_model: str = "YOUR_INTERNAL_MODEL_NAME"
    judge_api_key: str = "YOUR_API_KEY"
    judge_api_timeout: int = 30       # seconds per call
    judge_max_workers: int = 16       # parallel threads for batch API calls

    # ------------------------------------------------------------------ #
    # Rule-based reward bounds (tokens, derived from your data stats)     #
    # p10 ≈ 20 tokens, p90 ≈ 180 tokens based on SFT data description.   #
    # ------------------------------------------------------------------ #
    min_completion_tokens: int = 20
    max_completion_tokens: int = 200

    # ------------------------------------------------------------------ #
    # GRPO / training hyperparameters                                     #
    # ------------------------------------------------------------------ #
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2    # prompts per GPU per step
    gradient_accumulation_steps: int = 4   # effective batch = 2×8GPUs×4 = 64 prompts
    num_generations: int = 8               # G completions per prompt (group size)
    max_new_tokens: int = 200              # max tokens to generate per completion
    # Prompt is truncated to this length; 896 - 200 = 696 → round down for safety
    max_prompt_length: int = 700
    beta: float = 0.01                     # KL penalty coefficient vs. reference
    learning_rate: float = 1e-5            # lower than SFT (5e-5)
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    logging_steps: int = 10
    save_steps: int = 100
    seed: int = 42

    def get_grpo_config(self) -> GRPOConfig:
        """Build and return a trl.GRPOConfig from this config."""
        return GRPOConfig(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_generations=self.num_generations,
            max_new_tokens=self.max_new_tokens,
            max_prompt_length=self.max_prompt_length,
            beta=self.beta,
            learning_rate=self.learning_rate,
            lr_scheduler_type=self.lr_scheduler_type,
            warmup_ratio=self.warmup_ratio,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            seed=self.seed,
            bf16=True,
            gradient_checkpointing=True,
            dataloader_num_workers=4,
            # Keep all dataset columns so reward fns can access _meta etc.
            remove_unused_columns=False,
            # Log reward components separately to wandb
            report_to="wandb",
            # reward_weights is passed directly to GRPOTrainer, not GRPOConfig.
            # See rl_train.py.
        )
