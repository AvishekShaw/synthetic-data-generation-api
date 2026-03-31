#!/usr/bin/env python3
"""
generate_conversations.py

Generates synthetic UserLM training conversations for financial chatbot domain.
Loops over persona × scenario combinations, uses real conversations as few-shot
examples, and calls either the Anthropic API or a Databricks serving endpoint.

Usage:
    # Anthropic API (from your laptop)
    python generate_conversations.py \
        --provider anthropic \
        --api-key sk-ant-... \
        --data-path ~/Desktop/userlm_data.jsonl \
        --output synthetic_conversations.jsonl

    # Databricks endpoint
    python generate_conversations.py \
        --provider databricks \
        --databricks-host https://xxx.azuredatabricks.net \
        --databricks-token dapiXXX \
        --model your-endpoint-name \
        --data-path ~/Desktop/userlm_data.jsonl \
        --output synthetic_conversations.jsonl

    # Dry run: print prompts only, no API calls
    python generate_conversations.py \
        --provider dry_run \
        --data-path ~/Desktop/userlm_data.jsonl \
        --output prompts_preview.txt
"""

import json
import os
import re
import time
import random
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class Persona:
    knowledge_level: str       # novice | intermediate | expert
    emotional_state: str       # calm | mildly_frustrated | escalating
    communication_style: str   # direct | indirect | terse | verbose
    goal_clarity: str          # clear | vague | wrong_mental_model


@dataclass
class Scenario:
    # Structural: determines conversation path
    certainty: str                  # certain_fraud | uncertain
    information_completeness: str   # full | partial | minimal
    prior_contact: str              # first_contact | tried_merchant | follow_up
    expected_resolution: str        # refund_and_card | investigation | hold
    # Content: fills in the details
    merchant_name: str              # e.g. "Amazon", "Netflix"
    amount: str                     # e.g. "$127.43"
    transaction_date: str           # e.g. "August 20th"


# ─────────────────────────────────────────────
# DIMENSION DEFINITIONS
# ─────────────────────────────────────────────

# 6 representative persona types — diverse enough to avoid collapse
# without being so fine-grained that combinations explode
PERSONAS = [
    # ── Novice ──────────────────────────────────────────────────────────
    # Persona("novice",        "calm",              "indirect", "vague"),
    Persona("novice",        "mildly_frustrated", "direct",   "clear"),
    # Persona("novice",        "escalating",        "verbose",  "clear"),

    # ── Intermediate ────────────────────────────────────────────────────
    Persona("intermediate",  "calm",              "terse",    "clear"),
    Persona("intermediate",  "calm",              "indirect", "wrong_mental_model"),  # knows basics but thinks they need a police report / extra steps first
    Persona("intermediate",  "mildly_frustrated", "direct",   "vague"),
    Persona("intermediate",  "mildly_frustrated", "terse",    "clear"),               # been waiting, just wants it done, minimal words
    Persona("intermediate",  "escalating",        "direct",   "clear"),               # knows enough to push back hard when process feels slow

    # ── Expert ──────────────────────────────────────────────────────────
    Persona("expert",        "calm",              "direct",   "clear"),
    Persona("expert",        "calm",              "terse",    "clear"),               # Reg E aware, just files the dispute efficiently
    Persona("expert",        "mildly_frustrated", "direct",   "clear"),               # has done this before, impatient with standard verification steps
    Persona("expert",        "escalating",        "direct",   "clear"),               # threatens CFPB complaint, knows provisional credit timelines
]

# 8 structural scenario templates covering distinct conversation paths.
# Content (merchant, amount, date) is varied across templates to add diversity.
SCENARIOS = [
    # 1. Clean dispute — has everything, first contact, wants full resolution
    Scenario("certain_fraud", "full",    "first_contact",   "refund_and_card",
             "Amazon",        "$127.43", "August 20th"),
    # 2. Partial info — knows it's fraud, missing merchant details
    Scenario("certain_fraud", "partial", "first_contact",   "refund_and_card",
             "Shell Station", "$89.21",  "August 18th"),
    # 3. Already tried merchant — frustrated from the start
    Scenario("certain_fraud", "full",    "tried_merchant",  "refund_and_card",
             "TechHub",       "$340.00", "August 15th"),
    # 4. Uncertain — not sure if they authorized it (forgotten subscription?)
    Scenario("uncertain",     "partial", "first_contact",   "investigation",
             "Netflix",       "$34.99",  "August 12th"),
    # 5. Minimal info — just noticed something odd, wants card frozen first
    Scenario("certain_fraud", "minimal", "first_contact",   "hold",
             "Unknown",       "$300.00", "August 17th"),
    # 6. Follow-up — already called bank, no response, calling back
    Scenario("certain_fraud", "full",    "follow_up",       "refund_and_card",
             "Uber Eats",     "$215.00", "August 10th"),
    # 7. High stakes — large amount, wants immediate action
    Scenario("certain_fraud", "full",    "first_contact",   "refund_and_card",
             "Apple Store",   "$1,249.00","August 19th"),
    # 8. Uncertain with full info — has the details but genuinely not sure
    Scenario("uncertain",     "full",    "first_contact",   "investigation",
             "Adobe",         "$54.99",  "August 14th"),
]


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def _parse_line(line: str) -> Optional[dict]:
    """Parse a JSONL line, fixing common invalid escape sequences."""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        # Some entries have \$ or other non-standard escapes — fix them
        fixed = re.sub(r'\\([^"\\/bfnrtu])', lambda m: '\\\\' + m.group(1), line)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return None


def _extract_conversation(entry: dict) -> Optional[str]:
    """
    From an entry where output == '<eos>', extract the full conversation
    text from the 'Conversation so far:' section of the input.
    """
    inp = entry.get("input", "")
    if "Conversation so far:" in inp:
        convo = inp.split("Conversation so far:")[1]
        convo = convo.split("If your goal")[0].strip()
        return convo if convo else None
    # Some eos entries are the very first turn — no history yet
    # In that case, the output itself was the user's opening message,
    # which is stored in the entry that PRECEDES the eos entry.
    # We skip these (too short to be useful few-shot examples).
    return None


def load_few_shot_examples(data_path: str) -> list[str]:
    """
    Load complete conversations from the fine-tuning JSONL file.
    Extracts entries where output == '<eos>' and reconstructs conversations.
    Returns list of conversation strings in 'Customer: / Agent:' format.
    """
    examples = []
    path = Path(data_path).expanduser()

    with open(path, "rb") as f:
        for line in f:
            raw = line.decode("utf-8", errors="replace").strip()
            if not raw:
                continue
            entry = _parse_line(raw)
            if entry is None:
                continue
            if str(entry.get("output", "")).strip() == "<eos>":
                convo = _extract_conversation(entry)
                if convo:
                    examples.append(convo)

    print(f"  Loaded {len(examples)} complete conversations as few-shot examples.")
    return examples


# ─────────────────────────────────────────────
# PROMPT CONSTRUCTION
# ─────────────────────────────────────────────

def _persona_description(p: Persona) -> str:
    knowledge = {
        "novice": (
            "Does NOT know financial terminology. Says 'weird charge', 'that company', "
            "'my account' — not 'unauthorized transaction' or 'chargeback'. "
            "May not know what a dispute process involves."
        ),
        "intermediate": (
            "Knows basic terms (statement, transaction, dispute) but not advanced ones "
            "(provisional credit, chargeback timeframe). Has dealt with banks before."
        ),
        "expert": (
            "Comfortable with financial terminology. May mention chargeback rights, "
            "Regulation E, or provisional credit timelines. Knows the process."
        ),
    }
    emotional = {
        "calm": (
            "Starts matter-of-factly. Gets mildly impatient only if the process "
            "drags on unnecessarily."
        ),
        "mildly_frustrated": (
            "Already a bit annoyed before the chat starts. "
            "Uses 'look', sighs in text, says things like 'I already told you'."
        ),
        "escalating": (
            "Starts frustrated and escalates quickly. Threatens to close account "
            "or contact consumer protection if resolution takes too long."
        ),
    }
    style = {
        "direct":   "Gets to the point fast. Short messages. Doesn't over-explain.",
        "indirect": "Takes a few turns to get to the real issue. Provides lots of context first.",
        "terse":    "Very short messages — often 3-6 words. Abrupt. Minimal punctuation.",
        "verbose":  "Long messages. Over-explains. Repeats themselves. Lots of detail.",
    }
    clarity = {
        "clear":              "Knows exactly what they want: refund, new card, investigation opened.",
        "vague":              "Not sure what resolution they want. Just wants 'this sorted'. Needs the agent to guide them.",
        "wrong_mental_model": "Has a slightly wrong understanding of how disputes work — e.g., thinks they need to prove fraud before anything can happen.",
    }
    return (
        f"- Financial knowledge: {knowledge[p.knowledge_level]}\n"
        f"- Emotional state: {emotional[p.emotional_state]}\n"
        f"- Communication style: {style[p.communication_style]}\n"
        f"- Goal clarity: {clarity[p.goal_clarity]}"
    )


def _scenario_description(s: Scenario) -> str:
    certainty = {
        "certain_fraud": "CERTAIN they did not make this transaction. No ambiguity.",
        "uncertain":     "NOT SURE if they authorized it — could be a forgotten subscription or family member.",
    }
    info = {
        "full":    f"Has all details: merchant ({s.merchant_name}), amount ({s.amount}), date ({s.transaction_date}).",
        "partial": f"Knows roughly when ({s.transaction_date}) and approximately how much ({s.amount}), but can't recall the exact merchant name.",
        "minimal": "Just noticed 'something that looks wrong' on their statement. Doesn't have the details in front of them.",
    }
    contact = {
        "first_contact":  "First attempt to resolve this. Hasn't tried anything yet.",
        "tried_merchant": f"Already contacted {s.merchant_name} directly. Merchant refused to help. Now trying the bank.",
        "follow_up":      "Already called the bank once. Was told to wait. No update received. Following up now.",
    }
    resolution = {
        "refund_and_card": "Wants: (1) money back, (2) card cancelled and replaced.",
        "investigation":   "Just wants someone to investigate. Not demanding a refund yet.",
        "hold":            "Wants the card frozen immediately as a first step.",
    }
    return (
        f"- Dispute certainty: {certainty[s.certainty]}\n"
        f"- Information available: {info[s.information_completeness]}\n"
        f"- Prior contact: {contact[s.prior_contact]}\n"
        f"- What the customer wants: {resolution[s.expected_resolution]}"
    )


SYSTEM_PROMPT = """You are generating synthetic training data for a banking chatbot user simulator.

Write a realistic multi-turn conversation between a CUSTOMER and a BANKING AGENT.
The customer is dealing with a potentially fraudulent transaction.

CRITICAL RULES — based on real customer behavior:
1. Customer messages are SHORT and informal. Like text messages, not emails.
2. Customers don't use financial jargon unless they're financially sophisticated.
3. Customers give INCOMPLETE information first, add details only when prompted.
4. Customers ask about ONE thing at a time before moving to the next.
5. Customers REPEAT their request in different words when the agent doesn't understand.
6. Customers express frustration INDIRECTLY before directly.
7. Customers sometimes don't know what they actually need ("just want this sorted").
8. Include realistic typos, lowercase, skipped punctuation — but don't overdo it.
9. Customer messages are typically 1-2 sentences. Rarely more.

The agent is a banking chatbot that:
- Lists card options when needed (e.g. "Chase Freedom ****1234, Chase Sapphire ****5678")
- Lists recent transactions when needed
- Asks clarifying questions one at a time
- Is polite but slightly robotic/formulaic

FORMAT — alternate strictly between Customer and Agent:
Customer: [message]
Agent: [message]
Customer: [message]
...

Write 6-14 turns total. End when the customer's goal is complete or they've been redirected."""


def build_prompt(persona: Persona, scenario: Scenario, few_shot_examples: list[str]) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for this persona × scenario combination."""
    # Pick 3 random few-shot examples
    samples = random.sample(few_shot_examples, min(3, len(few_shot_examples)))
    few_shot_block = "\n\n---\n\n".join(samples)

    user_prompt = f"""Here are real examples of how actual banking customers chat with support agents:

=== REAL EXAMPLES ===
{few_shot_block}
=== END EXAMPLES ===

Now generate a NEW conversation with these specifications:

CUSTOMER PROFILE:
{_persona_description(persona)}

SCENARIO:
{_scenario_description(scenario)}

Generate the conversation now. Customer messages must sound like real people texting — not formal writing."""

    return SYSTEM_PROMPT, user_prompt


# ─────────────────────────────────────────────
# API CLIENTS
# ─────────────────────────────────────────────

def call_anthropic(system_prompt: str, user_prompt: str,
                   api_key: str, model: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        temperature=1.0,   # Higher temp = more diverse outputs
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text


def call_databricks(system_prompt: str, user_prompt: str,
                    host: str, token: str, model: str) -> str:
    """
    Calls a Databricks Model Serving endpoint via the OpenAI-compatible API.
    The endpoint must support chat completions format.
    """
    from openai import OpenAI
    client = OpenAI(
        api_key=token,
        base_url=f"{host.rstrip('/')}/serving-endpoints",
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=2048,
        temperature=1.0,
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────
# OUTPUT PARSING
# ─────────────────────────────────────────────

def parse_conversation(raw: str) -> list[dict]:
    """
    Parse 'Customer: ... \n Agent: ...' format into a list of
    {"role": "user"|"assistant", "content": "..."} dicts.
    Also handles multi-part agent responses (consecutive Agent: lines).
    """
    messages = []
    current_role = None
    current_lines = []

    def flush():
        if current_role and current_lines:
            messages.append({
                "role": current_role,
                "content": " ".join(current_lines).strip()
            })

    for line in raw.strip().split("\n"):
        if line.startswith("Customer:"):
            flush()
            current_role = "user"
            current_lines = [line[len("Customer:"):].strip()]
        elif line.startswith("Agent:"):
            # If previous was also agent, merge (multi-part agent response)
            if current_role == "assistant":
                current_lines.append(line[len("Agent:"):].strip())
            else:
                flush()
                current_role = "assistant"
                current_lines = [line[len("Agent:"):].strip()]
        elif line.strip() and current_role:
            current_lines.append(line.strip())

    flush()
    return messages


def generate_intent_summary(scenario: Scenario) -> str:
    """
    Constructs a scenario-specific intent summary in the style of the
    scenario_generation_prompt.  Fills the [INTENT] slot in both
    FIRST_TURN_TEMPLATE and COMPLETION_TEMPLATE.
    """
    action = {
        "certain_fraud": "report and dispute a fraudulent charge",
        "uncertain":     "investigate a suspicious charge",
    }[scenario.certainty]

    info = {
        "full":    (f"of {scenario.amount} at {scenario.merchant_name} "
                    f"on {scenario.transaction_date}"),
        "partial": (f"of approximately {scenario.amount} around "
                    f"{scenario.transaction_date}"),
        "minimal": "that appeared on your bank statement",
    }[scenario.information_completeness]

    resolution = {
        "refund_and_card": "get a refund and have your card cancelled and replaced",
        "investigation":   "have the bank open a formal investigation into the charge",
        "hold":            "get your card frozen immediately to prevent further charges",
    }[scenario.expected_resolution]

    prior = {
        "first_contact":  "",
        "tried_merchant": (f" You already contacted {scenario.merchant_name} "
                           f"directly but they refused to help."),
        "follow_up":      (" You already spoke with the bank once and are "
                           "following up after receiving no update."),
    }[scenario.prior_contact]

    return (
        f"You are a user chatting with an assistant language model to "
        f"{action} {info} and {resolution}.{prior}"
    )


# Training prompt templates — kept as module-level constants so they're
# easy to inspect and update without touching the logic.
# [INTENT] and [CONVERSATION HISTORY] are filled in at generation time.
FIRST_TURN_TEMPLATE = (
    "You are a human user interacting with an AI system. {intent}.\n"
    "Users can make typos, they don't always use perfect punctuation, "
    "and they tend to be lazy because typing requires effort.\n"
    "You have to also split information across turns and not give "
    "everything at the start.\n"
    "However, you should not overdo these things in your outputs, "
    "you must realistically act like a human.\n"
    "Generate the first prompt you would say to the system to achieve your goal."
)

COMPLETION_TEMPLATE = (
    "You are a human user interacting with an AI system. {intent}.\n"
    "Users can make typos, they don't always use perfect punctuation, "
    "and they tend to be lazy because typing requires effort.\n"
    "You have to also split information across turns and not give "
    "everything at the start.\n"
    "However, you should not overdo these things in your outputs, "
    "you must realistically act like a human.\n\n"
    "Conversation so far:\n"
    "{history}\n\n"
    "If your goal is complete, respond with <eos>. "
    "Otherwise, generate your next message."
)


def to_training_format(messages: list[dict], persona: Persona,
                       scenario: Scenario, idx: int) -> list[dict]:
    """
    Convert a generated conversation into the training JSONL format.
    One entry per user turn:
      - First user turn  → FIRST_TURN_TEMPLATE  (no history yet)
      - All later turns  → COMPLETION_TEMPLATE  (includes history so far)
    The last user turn always has output = '<eos>'.

    The [INTENT] slot is filled with a scenario-specific summary derived
    from the scenario dimensions (merchant, amount, date, certainty, etc.)
    rather than the generic "dispute a potentially fraudulent transaction".
    """
    intent = generate_intent_summary(scenario)

    entries = []
    history_lines: list[str] = []

    for i, msg in enumerate(messages):
        if msg["role"] == "user":
            is_last_user_turn = not any(
                m["role"] == "user" for m in messages[i+1:]
            )
            output = "<eos>" if is_last_user_turn else msg["content"] + "<eos>"

            if not history_lines:
                # First turn — no conversation history yet
                input_text = FIRST_TURN_TEMPLATE.format(intent=intent)
            else:
                # Completion turn — all middle and final turns
                input_text = COMPLETION_TEMPLATE.format(
                    intent=intent,
                    history="\n".join(history_lines),
                )

            entries.append({
                "input":  input_text,
                "output": output,
                "_meta": {
                    "synthetic":      True,
                    "generation_idx": idx,
                    "persona":        asdict(persona),
                    "scenario":       asdict(scenario),
                },
            })
            history_lines.append(f"Customer: {msg['content']}")

        elif msg["role"] == "assistant":
            history_lines.append(f"Agent: {msg['content']}")

    return entries


# ─────────────────────────────────────────────
# HUMAN-READABLE DUMP
# ─────────────────────────────────────────────

def write_conversation_dump(
    dump_dir: Path,
    idx: int,
    persona: Persona,
    scenario: Scenario,
    messages: list[dict],
) -> Path:
    """
    Write one human-readable txt file for a generated conversation.
    File name: conversation_NNN.txt  (zero-padded to 3 digits).
    Content: metadata header + full Customer/Agent dialogue.
    """
    dump_dir.mkdir(parents=True, exist_ok=True)
    out = dump_dir / f"conversation_{idx+1:03d}.txt"

    lines = []
    lines.append("=" * 60)
    lines.append(f"CONVERSATION {idx+1}")
    lines.append("=" * 60)
    lines.append("")
    lines.append("PERSONA:")
    lines.append(f"  knowledge_level    : {persona.knowledge_level}")
    lines.append(f"  emotional_state    : {persona.emotional_state}")
    lines.append(f"  communication_style: {persona.communication_style}")
    lines.append(f"  goal_clarity       : {persona.goal_clarity}")
    lines.append("")
    lines.append("SCENARIO:")
    lines.append(f"  certainty              : {scenario.certainty}")
    lines.append(f"  information_completeness: {scenario.information_completeness}")
    lines.append(f"  prior_contact          : {scenario.prior_contact}")
    lines.append(f"  expected_resolution    : {scenario.expected_resolution}")
    lines.append(f"  merchant               : {scenario.merchant_name}")
    lines.append(f"  amount                 : {scenario.amount}")
    lines.append(f"  transaction_date       : {scenario.transaction_date}")
    lines.append("")
    lines.append("-" * 60)
    lines.append("")

    for msg in messages:
        speaker = "Customer" if msg["role"] == "user" else "Agent"
        lines.append(f"{speaker}: {msg['content']}")
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ─────────────────────────────────────────────
# MAIN GENERATION LOOP
# ─────────────────────────────────────────────

def generate_all(args):
    print("\n=== UserLM Synthetic Data Generator ===\n")

    # Load few-shot examples
    print(f"Loading few-shot examples from: {args.data_path}")
    few_shot = load_few_shot_examples(args.data_path)
    if not few_shot:
        print("ERROR: No complete conversations found. Check your data path.")
        return

    # Build combinations
    combinations = [(p, s) for p in PERSONAS for s in SCENARIOS]
    total_available = len(combinations)

    # --limit controls how many to generate.
    # Default is 3 (for a quick test run).
    # Pass --limit all (or --limit -1) to generate the full set.
    if args.limit == -1:
        limit = total_available
        limit_label = "all"
    else:
        limit = min(args.limit, total_available)
        limit_label = str(limit)

    print(f"\nGenerating {limit_label} of {total_available} conversations "
          f"({len(PERSONAS)} personas × {len(SCENARIOS)} scenarios)")
    if limit < total_available:
        print(f"  ⚠ Test mode: only generating {limit} conversations. "
              f"Pass --limit all to generate the full set.")
    print(f"Provider: {args.provider}")
    print(f"Output: {args.output}\n")

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Human-readable dumps go in a sibling directory
    dump_dir = output_path.parent / "conversation_dumps"

    # Track progress — skip already-generated combinations if resuming
    generated_indices = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    idx = entry.get("_meta", {}).get("generation_idx")
                    if idx is not None:
                        generated_indices.add(idx)
                except:
                    pass
        if generated_indices:
            print(f"  Resuming — {len(generated_indices)} combinations already done.\n")

    dry_run_path = None
    if args.provider == "dry_run":
        dry_run_path = output_path.with_suffix(".prompts.txt")
        dry_run_file = open(dry_run_path, "w")

    success_count = 0
    error_count = 0

    for idx, (persona, scenario) in enumerate(combinations):
        # Stop once we've hit the limit (counting only new successes)
        if success_count >= limit:
            remaining = total_available - idx
            print(f"\n  Limit of {limit} reached. "
                  f"{remaining} combination(s) remaining — run with --limit all to generate everything.")
            break

        if idx in generated_indices:
            print(f"  [{idx+1}/{total_available}] Skipping (already done)")
            continue

        label = (f"[{idx+1}/{total_available}] "
                 f"{persona.knowledge_level}/{persona.emotional_state} | "
                 f"{scenario.certainty}/{scenario.information_completeness} | "
                 f"{scenario.merchant_name} {scenario.amount}")
        print(label)

        system_prompt, user_prompt = build_prompt(persona, scenario, few_shot)

        # ── Dry run: just save the prompts ──
        if args.provider == "dry_run":
            dry_run_file.write(f"\n{'='*60}\n")
            dry_run_file.write(f"COMBINATION {idx+1}/{total_available}\n")
            dry_run_file.write(f"Persona: {asdict(persona)}\n")
            dry_run_file.write(f"Scenario: {asdict(scenario)}\n")
            dry_run_file.write(f"{'='*60}\n\n")
            dry_run_file.write(f"SYSTEM:\n{system_prompt}\n\n")
            dry_run_file.write(f"USER:\n{user_prompt}\n\n")
            print(f"  → Prompt saved")
            success_count += 1   # count dry-run prompts against the limit too
            continue

        # ── API call with retry ──
        raw = None
        for attempt in range(3):
            try:
                if args.provider == "anthropic":
                    raw = call_anthropic(
                        system_prompt, user_prompt,
                        args.api_key, args.model
                    )
                elif args.provider == "databricks":
                    raw = call_databricks(
                        system_prompt, user_prompt,
                        args.databricks_host, args.databricks_token, args.model
                    )
                break
            except Exception as e:
                wait = 2 ** attempt * 5
                print(f"  Attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)

        if raw is None:
            print(f"  ✗ All retries failed — skipping.")
            error_count += 1
            continue

        # ── Parse and save ──
        messages = parse_conversation(raw)
        if len(messages) < 2:
            print(f"  ✗ Generated conversation too short ({len(messages)} turns) — skipping.")
            error_count += 1
            continue

        training_entries = to_training_format(messages, persona, scenario, idx)

        with open(output_path, "a") as f:
            for entry in training_entries:
                f.write(json.dumps(entry) + "\n")

        # Write human-readable dump for this conversation
        dump_path = write_conversation_dump(dump_dir, idx, persona, scenario, messages)

        user_turns = sum(1 for m in messages if m["role"] == "user")
        print(f"  ✓ {len(messages)} turns ({user_turns} user) → {len(training_entries)} training examples")
        print(f"    Dump: {dump_path}")
        success_count += 1

        # Rate limiting
        if args.provider == "anthropic":
            time.sleep(0.5)

    # ── Summary ──
    if args.provider == "dry_run":
        dry_run_file.close()
        print(f"\nDone. Prompts saved to: {dry_run_path}")
        print("Copy-paste individual prompts into Claude.ai to generate manually.")
    else:
        print(f"\n{'='*40}")
        print(f"Done. {success_count} conversations generated, {error_count} failed.")
        print(f"Output JSONL : {output_path}")
        print(f"Readable dumps: {dump_dir}/")

        # Quick stats on output
        try:
            eos_count = 0
            total_entries = 0
            with open(output_path) as f:
                for line in f:
                    e = json.loads(line)
                    total_entries += 1
                    if e.get("output") == "<eos>":
                        eos_count += 1
            print(f"Total training entries written: {total_entries}")
            print(f"  of which complete conversations (<eos>): {eos_count}")
        except:
            pass


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic UserLM training conversations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "databricks", "dry_run"],
        required=True,
        help=(
            "anthropic: Anthropic API (needs --api-key). "
            "databricks: Databricks serving endpoint. "
            "dry_run: save prompts to file, no API calls."
        ),
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to userlm_data.jsonl (your existing fine-tuning data).",
    )
    parser.add_argument(
        "--output",
        default="synthetic_conversations.jsonl",
        help="Output JSONL file (same format as input data, ready for fine-tuning).",
    )

    parser.add_argument(
        "--limit",
        default=3,
        help=(
            "How many conversations to generate. "
            "Default is 3 (test run). "
            "Pass 'all' (or -1) to generate the full persona × scenario set."
        ),
    )

    # Anthropic options
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ANTHROPIC_API_KEY"),
        help="Anthropic API key. Defaults to ANTHROPIC_API_KEY env var.",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Model name for Anthropic (default: claude-sonnet-4-6) or endpoint name for Databricks.",
    )

    # Databricks options
    parser.add_argument(
        "--databricks-host",
        default=os.environ.get("DATABRICKS_HOST"),
        help="Databricks workspace URL, e.g. https://xxx.azuredatabricks.net. "
             "Defaults to DATABRICKS_HOST env var.",
    )
    parser.add_argument(
        "--databricks-token",
        default=os.environ.get("DATABRICKS_TOKEN"),
        help="Databricks personal access token. Defaults to DATABRICKS_TOKEN env var.",
    )

    args = parser.parse_args()

    # Normalise --limit: "all" or "-1" → sentinel -1, otherwise parse as int
    if str(args.limit).lower() == "all":
        args.limit = -1
    else:
        try:
            args.limit = int(args.limit)
            if args.limit < 1 and args.limit != -1:
                parser.error("--limit must be a positive integer, -1, or 'all'.")
        except ValueError:
            parser.error(f"--limit: invalid value '{args.limit}'. Use a number or 'all'.")

    # Validation
    if args.provider == "anthropic" and not args.api_key:
        parser.error("--api-key required for anthropic provider "
                     "(or set ANTHROPIC_API_KEY env var).")
    if args.provider == "databricks":
        if not args.databricks_host:
            parser.error("--databricks-host required (or set DATABRICKS_HOST env var).")
        if not args.databricks_token:
            parser.error("--databricks-token required (or set DATABRICKS_TOKEN env var).")

    generate_all(args)


if __name__ == "__main__":
    main()
