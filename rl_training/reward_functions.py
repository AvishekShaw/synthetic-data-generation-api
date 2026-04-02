"""
reward_functions.py
-------------------
Two reward functions for GRPO training of the UserLM:

  1. judge_reward_fn    — LLM-as-judge scoring via internal API.
                          Scores 5 dimensions of customer message quality.
                          Returns a scalar in [0, 1].

  2. rule_based_reward_fn — Fast, deterministic checks (length, EOS, repetition).
                            Returns a scalar in [0, 1].

Both follow TRL's reward function signature:
    fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]

kwargs contains any additional columns from your dataset (e.g. generation_idx,
reference) — useful for logging / debugging but not required for scoring.

Reward weighting is handled in rl_train.py via GRPOTrainer(reward_weights=[...]).
TRL will log each function's scores separately to wandb.
"""

from __future__ import annotations
import json
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Judge prompts
# ══════════════════════════════════════════════════════════════════════════════

_JUDGE_SYSTEM = """\
You are an expert evaluator assessing the quality of synthetic customer messages \
in banking fraud dispute conversations.

You will be shown:
  1. The conversation context (system persona + dialogue history up to this point)
  2. A candidate customer message to evaluate

Your job is to score the message on 5 dimensions. Be strict and calibrated — \
a score of 3 means genuinely excellent, not merely acceptable."""

_JUDGE_RUBRIC = """\
## Scoring Dimensions

**1. Persona Consistency (0–3)**
Does the customer's vocabulary, tone, emotional baseline, and financial sophistication \
match the persona described in the system prompt?
  0 — Clearly wrong register or demographics
  1 — Mostly in character with noticeable lapses
  2 — Consistently in character throughout
  3 — Exemplary embodiment; distinctive, fully believable

**2. Contextual Coherence (0–3)**
Does the message directly address what the bank agent just said?
  0 — Ignores the agent or is a non-sequitur
  1 — Partially responds to the agent
  2 — Clearly and directly responds
  3 — Responds naturally and advances the conversation meaningfully

**3. Emotional Realism (0–3)**
Is the emotional state appropriate to both the fraud situation and the conversation arc?
  0 — Wrong register (e.g., oddly calm right after discovering fraud)
  1 — Correct emotion but feels performed or exaggerated
  2 — Natural emotional state
  3 — Nuanced; emotion evolves believably from the conversation history

**4. Information Pacing (0–3)**
Does the customer reveal transaction details at a realistic rate?
  0 — Dumps all information at once, or withholds info they clearly have
  1 — Mostly natural pacing with one clear error
  2 — Naturally paced throughout
  3 — Perfect pacing — provides exactly what a real customer would at this moment

**5. Conversational Naturalness (0–2)**
Does this read like a real person typing, not an AI generating text?
  0 — Clearly AI-generated (over-formal, no hedging, suspiciously complete)
  1 — Mostly natural with some stiffness
  2 — Indistinguishable from a real customer message

## Output Format
Respond with ONLY valid JSON, no markdown fences:
{
  "persona_consistency": <int 0-3>,
  "contextual_coherence": <int 0-3>,
  "emotional_realism": <int 0-3>,
  "information_pacing": <int 0-3>,
  "conversational_naturalness": <int 0-2>,
  "reason": "<one concise sentence>"
}"""

_MAX_SCORE = 3 + 3 + 3 + 3 + 2  # = 14


# ══════════════════════════════════════════════════════════════════════════════
# Internal API call (placeholder — swap for your actual client)
# ══════════════════════════════════════════════════════════════════════════════

def _call_judge_api(
    prompt: str,
    completion: str,
    api_url: str,
    model: str,
    api_key: str,
    timeout: int = 30,
) -> Optional[dict]:
    """
    Call internal LLM endpoint and parse JSON scores.

    Returns a dict with keys matching the rubric dimensions, or None on failure.
    Replace the requests.post body if your internal API has a different schema.
    """
    user_message = (
        f"## Conversation Context\n{prompt}\n\n"
        f"## Candidate Customer Message\n{completion}\n\n"
        f"{_JUDGE_RUBRIC}"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user",   "content": user_message},
        ],
        "max_tokens": 256,
        "temperature": 0.0,
    }

    try:
        resp = requests.post(
            api_url,
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()

        # Strip markdown fences if the model adds them despite instructions
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        return json.loads(raw)

    except requests.exceptions.Timeout:
        logger.warning("Judge API timed out for one completion; returning None.")
        return None
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Judge response parse error: {e}. Returning None.")
        return None
    except Exception as e:
        logger.warning(f"Judge API unexpected error: {e}. Returning None.")
        return None


def _scores_to_scalar(scores: Optional[dict]) -> float:
    """Normalise dimension scores to [0, 1]. Returns 0.0 on API failure."""
    if scores is None:
        return 0.0
    total = (
        scores.get("persona_consistency",      0)
        + scores.get("contextual_coherence",   0)
        + scores.get("emotional_realism",      0)
        + scores.get("information_pacing",     0)
        + scores.get("conversational_naturalness", 0)
    )
    return float(total) / _MAX_SCORE


# ══════════════════════════════════════════════════════════════════════════════
# Public reward functions (TRL-compatible signature)
# ══════════════════════════════════════════════════════════════════════════════

def make_judge_reward_fn(
    api_url: str,
    model: str,
    api_key: str,
    timeout: int = 30,
    max_workers: int = 16,
):
    """
    Factory that closes over API credentials and returns a TRL reward function.

    Usage:
        judge_reward_fn = make_judge_reward_fn(cfg.judge_api_url, ...)
        GRPOTrainer(reward_funcs=[judge_reward_fn, ...], ...)

    TRL calls each reward function once per training step with the full batch:
        fn(prompts, completions, **dataset_columns) -> list[float]

    We parallelise API calls with a thread pool (network-bound, not CPU-bound).
    """
    def judge_reward_fn(
        prompts: list[str],
        completions: list[str],
        **kwargs,        # extra dataset columns (generation_idx, reference, …)
    ) -> list[float]:
        # Build index-keyed futures so results are ordered correctly
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _call_judge_api, p, c, api_url, model, api_key, timeout
                ): i
                for i, (p, c) in enumerate(zip(prompts, completions))
            }
            scores = [0.0] * len(prompts)
            for future in as_completed(futures):
                idx = futures[future]
                scores[idx] = _scores_to_scalar(future.result())

        return scores

    judge_reward_fn.__name__ = "judge_reward"   # shows in wandb logs
    return judge_reward_fn


def make_rule_based_reward_fn(
    tokenizer,
    min_tokens: int = 20,
    max_tokens: int = 200,
):
    """
    Factory that closes over the tokenizer and length bounds.

    Three checks, each contributing a fraction of the total score:

      ┌──────────────────────────────┬────────┐
      │ Check                        │ Weight │
      ├──────────────────────────────┼────────┤
      │ EOS token present            │  0.40  │
      │ Length within [min, max]     │  0.40  │
      │ Low trigram self-repetition  │  0.20  │
      └──────────────────────────────┴────────┘

    Returns scores in [0, 1].
    """
    eos = tokenizer.eos_token  # e.g. "<eos>" for Gemma

    def rule_based_reward_fn(
        prompts: list[str],
        completions: list[str],
        **kwargs,
    ) -> list[float]:
        rewards = []

        for completion in completions:
            score = 0.0

            # ── 1. EOS token present ──────────────────────────────────────
            if eos in completion or "<eos>" in completion:
                score += 0.40

            # ── 2. Length within distribution bounds ─────────────────────
            token_ids = tokenizer.encode(completion, add_special_tokens=False)
            n_tokens = len(token_ids)
            if min_tokens <= n_tokens <= max_tokens:
                score += 0.40
            elif n_tokens < min_tokens:
                # Very short completions are usually degenerate — partial credit
                score += 0.10
            # Too long → no credit for length

            # ── 3. Trigram self-repetition check ─────────────────────────
            words = completion.lower().split()
            if len(words) < 3:
                # Too short to evaluate — give benefit of the doubt
                score += 0.20
            else:
                trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
                unique_ratio = len(set(trigrams)) / len(trigrams)
                if unique_ratio >= 0.80:
                    score += 0.20
                elif unique_ratio >= 0.50:
                    score += 0.10
                # Below 0.50 → highly repetitive, no credit

            rewards.append(score)

        return rewards

    rule_based_reward_fn.__name__ = "rule_reward"   # shows in wandb logs
    return rule_based_reward_fn
