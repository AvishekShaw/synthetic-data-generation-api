#!/usr/bin/env python3
"""
generate_probe_conversations.py

Generates Type A (inadvertent probing) and Type B (adversarial probing)
training conversations for the UserLM project.

Architecture: Alternating API calls — separate user model and agent model,
each with their own system prompt. This gives clean control over:
  - User behavioral instructions (info drip, prior beliefs, error awareness)
  - Agent planted errors (Type B only)

Type A — Inadvertent probing:
  The user behaves realistically but their natural behaviors (partial info,
  wrong prior beliefs, error awareness) stress the agent in realistic ways.

Type B — Adversarial probing:
  The agent is instructed to make a specific planted error. The user is
  instructed to push back on agent errors proportional to their persona.
  Trains the UserLM to catch and respond to real agent failure modes.

Usage:
    # Generate Type A conversations (dry run first)
    python generate_probe_conversations.py \
        --mode type_a --provider dry_run --limit 3

    # Generate Type A for real
    python generate_probe_conversations.py \
        --mode type_a --provider anthropic --api-key sk-ant-... --limit all

    # Generate Type B for real
    python generate_probe_conversations.py \
        --mode type_b --provider anthropic --api-key sk-ant-... --limit all

generation_idx offsets (to avoid collision with success conversations):
    success conversations : 0–999   (generate_conversations.py)
    type_a conversations  : 1000+
    type_b conversations  : 2000+
"""

import json
import os
import re
import sys
import time
import random
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path

# Allow running from repo root or from inside synthetic_data_generation/
sys.path.insert(0, str(Path(__file__).parent))
from content_pools import sample_content, sample_target_length  # noqa: E402


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
    certainty: str
    information_completeness: str
    prior_contact: str
    expected_resolution: str
    # Content: sampled per-conversation via content_pools.sample_content()
    merchant_name: str
    amount: str
    transaction_date: str
    card_type: str
    channel: str


@dataclass
class PlantedError:
    agent_failure_mode: str   # wrong_policy_fact | loop | contradiction | scope_failure | over_commit | wrong_escalation
    description: str          # what the agent should do wrong
    timing: str               # when to introduce the error


# ─────────────────────────────────────────────
# PERSONAS  (same set as generate_conversations.py)
# ─────────────────────────────────────────────

PERSONAS = [
    Persona("novice",        "calm",              "indirect", "vague"),
    Persona("novice",        "mildly_frustrated", "direct",   "clear"),
    Persona("novice",        "escalating",        "verbose",  "clear"),
    Persona("intermediate",  "calm",              "terse",    "clear"),
    Persona("intermediate",  "calm",              "indirect", "wrong_mental_model"),
    Persona("intermediate",  "calm",              "verbose",  "vague"),
    Persona("intermediate",  "mildly_frustrated", "direct",   "vague"),
    Persona("intermediate",  "mildly_frustrated", "terse",    "clear"),
    Persona("intermediate",  "escalating",        "direct",   "clear"),
    Persona("expert",        "calm",              "direct",   "clear"),
    Persona("expert",        "calm",              "terse",    "clear"),
    Persona("expert",        "mildly_frustrated", "direct",   "clear"),
    Persona("expert",        "escalating",        "direct",   "clear"),
    Persona("expert",        "escalating",        "verbose",  "clear"),
]

# 8 structural scenario templates — same as generate_conversations.py.
# Content fields (merchant_name, amount, transaction_date, card_type, channel)
# are left empty here and sampled per-conversation via content_pools.sample_content().
SCENARIOS = [
    Scenario("certain_fraud", "full",    "first_contact",  "refund_and_card",  "", "", "", "", ""),
    Scenario("certain_fraud", "partial", "first_contact",  "refund_and_card",  "", "", "", "", ""),
    Scenario("certain_fraud", "full",    "tried_merchant", "refund_and_card",  "", "", "", "", ""),
    Scenario("uncertain",     "partial", "first_contact",  "investigation",    "", "", "", "", ""),
    Scenario("certain_fraud", "minimal", "first_contact",  "hold",             "", "", "", "", ""),
    Scenario("certain_fraud", "full",    "follow_up",      "refund_and_card",  "", "", "", "", ""),
    Scenario("certain_fraud", "full",    "first_contact",  "refund_and_card",  "", "", "", "", ""),
    Scenario("uncertain",     "full",    "first_contact",  "investigation",    "", "", "", "", ""),
]


# ─────────────────────────────────────────────
# TYPE A: PRIOR BELIEFS POOL
# Flat pool of plausible misbeliefs — sampled independently per conversation.
# ─────────────────────────────────────────────

PRIOR_BELIEFS_POOL = [
    "You believe the dispute window is 120 days from the transaction date.",
    "You believe you can dispute any charge with just a rough description — no exact merchant name needed.",
    "You believe that since the merchant already refused to help, the bank is legally required to refund you immediately without further investigation.",
    "You believe that for uncertain charges, you need to be 100% sure it's fraud before the bank can do anything.",
    "You believe freezing your card happens instantly and automatically blocks all pending charges.",
    "You believe that as a returning caller with an open case, you are entitled to speak to a supervisor immediately.",
    "You believe that for charges over $500, federal law requires the bank to issue provisional credit within 24 hours.",
    "You believe you need to file a police report before the bank can open any dispute investigation.",
    "You believe that disputing a charge automatically gets your money back immediately while the investigation runs.",
    "You believe the bank can only dispute a charge if you have the exact transaction amount — approximate amounts aren't accepted.",
    "You believe that contacting the bank more than 30 days after the transaction means you've lost your right to dispute it.",
    "You believe that cancelling your card immediately wipes out any pending unauthorized charges on the account.",
]


# ─────────────────────────────────────────────
# TYPE B: PLANTED ERRORS POOL
# Flat pool — sampled independently per conversation.
# ─────────────────────────────────────────────

PLANTED_ERRORS_POOL = [
    PlantedError(
        "wrong_policy_fact",
        "State that the dispute window is 60 days from the transaction date. "
        "(The correct policy is 90 days.)",
        "On your first substantive response after the user explains their dispute.",
    ),
    PlantedError(
        "scope_failure",
        "State that without the exact merchant name you cannot open a dispute. "
        "(Incorrect — disputes can be opened with partial information.)",
        "When the user first mentions they don't have the exact merchant name.",
    ),
    PlantedError(
        "over_commit",
        "State that the refund will appear in their account within 24 hours. "
        "(Incorrect — investigation takes 3-5 business days minimum.)",
        "When you confirm the dispute has been opened.",
    ),
    PlantedError(
        "wrong_policy_fact",
        "State that the bank can only open a dispute if the customer is certain the "
        "charge is fraudulent — uncertain charges cannot be disputed. "
        "(Incorrect — investigation disputes for uncertain charges are fully valid.)",
        "On your second response after understanding the customer's uncertainty.",
    ),
    PlantedError(
        "contradiction",
        "First say the card freeze takes effect immediately. Then, two or more turns "
        "later, say that card freezes take 24 hours to process.",
        "Say 'immediately' early in the conversation; then contradict with '24 hours' "
        "after the customer has already acknowledged the freeze.",
    ),
    PlantedError(
        "loop",
        "Ask the customer for their full card number or account details even though "
        "they have already provided this information earlier in the conversation.",
        "After the customer has already given their card or account details, "
        "ask for them again as if you don't have them.",
    ),
    PlantedError(
        "wrong_escalation",
        "State that disputes over $1,000 cannot be processed via chat and require "
        "an in-branch visit. (Incorrect — all dispute amounts can be handled remotely.)",
        "When the transaction amount above $1,000 is first mentioned.",
    ),
    PlantedError(
        "wrong_policy_fact",
        "State that a police report is required before the bank can open a dispute "
        "for a potentially unauthorized charge. "
        "(Incorrect — a police report is optional, not required.)",
        "On your first substantive response after the customer explains the situation.",
    ),
    PlantedError(
        "wrong_policy_fact",
        "State that provisional credit is only available for transactions over $100. "
        "(Incorrect — provisional credit applies to all dispute amounts.)",
        "When the customer asks about getting their money back during the investigation.",
    ),
    PlantedError(
        "scope_failure",
        "State that you cannot process a dispute for a charge that occurred more than "
        "45 days ago. (Incorrect — the dispute window is 90 days.)",
        "After the customer mentions when the transaction occurred.",
    ),
    PlantedError(
        "over_commit",
        "State that a new card will arrive in 1-2 business days. "
        "(Incorrect — card replacement takes 5-7 business days.)",
        "When the customer asks about getting a replacement card.",
    ),
    PlantedError(
        "loop",
        "Ask the customer to re-verify their identity (name, date of birth, last 4 of SSN) "
        "even though they already completed verification earlier in the conversation.",
        "After at least 3 turns have passed since the customer verified their identity.",
    ),
]


# ─────────────────────────────────────────────
# PROMPT BUILDERS
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
            "Regulation E, or provisional credit timelines. Knows the process well."
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
        f"  Financial knowledge : {knowledge[p.knowledge_level]}\n"
        f"  Emotional state     : {emotional[p.emotional_state]}\n"
        f"  Communication style : {style[p.communication_style]}\n"
        f"  Goal clarity        : {clarity[p.goal_clarity]}"
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
        f"  Dispute certainty        : {certainty[s.certainty]}\n"
        f"  Information available    : {info[s.information_completeness]}\n"
        f"  Prior contact            : {contact[s.prior_contact]}\n"
        f"  What the customer wants  : {resolution[s.expected_resolution]}"
    )


def generate_intent_summary(scenario: Scenario) -> str:
    action = {
        "certain_fraud": "report and dispute a fraudulent charge",
        "uncertain":     "investigate a suspicious charge",
    }[scenario.certainty]
    info = {
        "full":    f"of {scenario.amount} at {scenario.merchant_name} on {scenario.transaction_date}",
        "partial": f"of approximately {scenario.amount} around {scenario.transaction_date}",
        "minimal": "that appeared on your bank statement",
    }[scenario.information_completeness]
    resolution = {
        "refund_and_card": "get a refund and have your card cancelled and replaced",
        "investigation":   "have the bank open a formal investigation into the charge",
        "hold":            "get your card frozen immediately to prevent further charges",
    }[scenario.expected_resolution]
    prior = {
        "first_contact":  "",
        "tried_merchant": f" You already contacted {scenario.merchant_name} directly but they refused to help.",
        "follow_up":      " You already spoke with the bank once and are following up after receiving no update.",
    }[scenario.prior_contact]
    return (
        f"You are a user chatting with an assistant language model to "
        f"{action} {info} and {resolution}.{prior}"
    )


def build_user_system_prompt(persona: Persona, scenario: Scenario,
                              mode: str, prior_belief: str,
                              target_length: int = 15) -> str:
    """Build the system prompt for the USER side (customer simulator)."""

    base = f"""You are simulating a realistic banking customer in a live chat dispute conversation.

PERSONA:
{_persona_description(persona)}

SCENARIO:
{_scenario_description(scenario)}

CORE BEHAVIORAL RULES:
1. Messages are informal — like texts or chat, not emails. Your messages tend to run about {target_length} words; vary naturally around that — shorter for quick replies, longer when explaining context.
2. Use language matching your financial knowledge level. Novices say "weird charge", experts say "unauthorized transaction".
3. Include occasional typos, lowercase, skipped punctuation — but don't overdo it.
4. Ask about ONE thing at a time. Do not give everything at once.
5. Express frustration indirectly first, then more directly if the issue isn't resolved.

INFORMATION DRIP:
Do not provide all transaction details in your first message. Give information gradually
as the agent asks for it. It is realistic to realize mid-conversation that you gave a
slightly wrong detail (e.g. wrong date, rounded amount) and correct yourself.

PRIOR BELIEF:
{prior_belief}
If the agent tells you something that contradicts this belief, react naturally and
proportionally to your communication style — push back, ask for clarification, or
express confusion. Do not immediately and silently accept what the agent says if it
conflicts with what you believe.

ERROR AWARENESS:
If the agent asks you a question you already answered earlier in this conversation,
notice it. React proportionally to your persona:
  - Terse: re-answer with no extra words, maybe a period at the end
  - Direct: briefly note you already said this ("already told you that")
  - Indirect/verbose: re-answer with mild confusion ("didn't I mention that? anyway...")
  - Escalating: show clear impatience ("I JUST said that")

CONSTRAINT REFINEMENT:
If the agent offers a resolution that only partially meets your stated goal, do not
accept a pivot to something else. Ask whether your specific goal is achievable before
settling for an alternative."""

    type_b_addition = ""
    if mode == "type_b":
        type_b_addition = """

ERROR RESPONSE (adversarial probing):
You may notice the agent stating something factually incorrect, contradicting itself,
or repeating a question already answered. When this happens, respond proportionally:
  - Direct/assertive  : name the error explicitly ("That's not right, I was told it's 90 days")
  - Expert            : cite your knowledge calmly ("Under Reg E, that's not correct")
  - Anxious/uncertain : ask for confirmation ("Are you sure about that? I thought...")
  - Terse             : state the correct information without elaboration
  - Escalating        : treat the error as further evidence of poor service

Do not let clear factual errors or contradictions pass without some reaction.
A real person notices when the agent says something that doesn't match what they know."""

    exit_instruction = """

EXIT CONDITION:
When your goal is fully complete, OR you have decided to give up / escalate to a
different channel, respond with exactly: <eos>
Do not include any other text alongside <eos>."""

    return base + type_b_addition + exit_instruction


def build_agent_system_prompt(mode: str, planted_error: "PlantedError | None" = None) -> str:
    """Build the system prompt for the AGENT side (banking chatbot simulator)."""
    base = """You are a banking dispute resolution chatbot. Be professional, polite, and concise.

YOUR CAPABILITIES:
- Open fraud dispute cases and assign case numbers
- Cancel and replace compromised cards (5-7 business days for new card)
- Issue provisional credit for clear fraud cases during investigation (within 2 business days)
- Investigate unauthorized charges (resolution in 3-5 business days)
- Freeze/unfreeze cards (takes effect immediately)
- View recent transactions and card details on file

CORRECT POLICY FACTS:
- Dispute window            : 90 days from transaction date
- All amounts handled remotely : no branch visit required for any dispute amount
- Partial info accepted     : disputes can be opened without exact merchant name
- Police report             : optional, never required
- Provisional credit        : typically within 2 business days of opening dispute
- Card freeze               : takes effect immediately
- Investigation resolution  : 3-5 business days

BEHAVIOR:
- Ask ONE clarifying question at a time — never stack multiple questions
- List card options when needed (e.g., "Visa ****1234, Mastercard ****5678")
- Be slightly formulaic — you are a chatbot, not a human agent
- Confirm key details before taking irreversible actions
- Always provide a case number when opening a dispute"""

    if mode == "type_b" and planted_error is not None:
        error = planted_error
        planted = f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRAINING DATA GENERATION INSTRUCTION (planted error):
You will make exactly ONE specific error during this conversation.

  ERROR TYPE : {error.agent_failure_mode}
  WHAT TO DO : {error.description}
  WHEN       : {error.timing}

After the customer responds to your error (whether or not they catch it),
continue the conversation normally toward resolution.
Do NOT spontaneously correct yourself — only correct if the customer
explicitly and insistently pushes back.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
        return base + planted

    return base


def build_turn_prompt(history: list[dict], role: str) -> str:
    """
    Build the prompt for a single turn.
    history: list of {"speaker": "Customer"|"Agent", "content": "..."}
    role: "user" (generate customer turn) | "agent" (generate agent turn)
    """
    if not history:
        if role == "user":
            return (
                "Generate the customer's opening message to start the conversation. "
                "Output ONLY the message text — no 'Customer:' prefix."
            )
        else:
            return (
                "The customer is about to send their first message. "
                "Output ONLY your agent response — no 'Agent:' prefix."
            )

    history_str = "\n".join(
        f"{turn['speaker']}: {turn['content']}" for turn in history
    )

    if role == "user":
        return (
            f"Conversation so far:\n{history_str}\n\n"
            "Generate ONLY the customer's next message. "
            "Output only the message text — no 'Customer:' prefix.\n"
            "If your goal is complete or you are giving up, output exactly: <eos>"
        )
    else:
        return (
            f"Conversation so far:\n{history_str}\n\n"
            "Generate ONLY the agent's next response. "
            "Output only the response text — no 'Agent:' prefix."
        )


# ─────────────────────────────────────────────
# API CLIENT
# ─────────────────────────────────────────────

def call_turn(system_prompt: str, turn_prompt: str,
              api_key: str, model: str) -> str:
    """Single-turn API call. Returns the model's text response."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=512,
        temperature=1.0,
        system=system_prompt,
        messages=[{"role": "user", "content": turn_prompt}],
    )
    return response.content[0].text.strip()


# ─────────────────────────────────────────────
# ALTERNATING CONVERSATION GENERATOR
# ─────────────────────────────────────────────

def generate_conversation_alternating(
    persona: Persona,
    scenario: Scenario,
    mode: str,
    args,
    prior_belief: str,
    planted_error: "PlantedError | None" = None,
    target_length: int = 15,
    max_turns: int = 14,
) -> tuple[list[dict], dict]:
    """
    Generate a full conversation via alternating user/agent API calls.

    Returns:
        messages  : list of {"role": "user"|"assistant", "content": str}
        probe_meta: dict of probing-specific _meta fields
    """
    user_system  = build_user_system_prompt(persona, scenario, mode, prior_belief, target_length)
    agent_system = build_agent_system_prompt(mode, planted_error)

    # history holds the running conversation for context injection
    history:  list[dict] = []   # {"speaker": "Customer"|"Agent", "content": str}
    messages: list[dict] = []   # training-format {"role": ..., "content": ...}

    goal_completed    = False
    user_caught_error = False

    for turn_num in range(max_turns):

        # ── CUSTOMER TURN ──────────────────────────────────────
        user_prompt = build_turn_prompt(history, "user")
        user_text   = None

        for attempt in range(3):
            try:
                user_text = call_turn(user_system, user_prompt, args.api_key, args.model)
                break
            except Exception as e:
                wait = 2 ** attempt * 3
                print(f"    [user turn attempt {attempt+1}] {e} — retrying in {wait}s")
                time.sleep(wait)

        if user_text is None:
            print("    ✗ User turn failed after 3 retries — ending conversation early.")
            break

        # Check for exit signal
        if user_text.strip() == "<eos>" or "<eos>" in user_text.lower():
            goal_completed = True
            break

        # Strip any accidental "Customer:" prefix the model may add
        user_text = re.sub(r"^Customer:\s*", "", user_text, flags=re.IGNORECASE).strip()

        history.append({"speaker": "Customer", "content": user_text})
        messages.append({"role": "user", "content": user_text})

        # ── AGENT TURN ─────────────────────────────────────────
        agent_prompt = build_turn_prompt(history, "agent")
        agent_text   = None

        for attempt in range(3):
            try:
                agent_text = call_turn(agent_system, agent_prompt, args.api_key, args.model)
                break
            except Exception as e:
                wait = 2 ** attempt * 3
                print(f"    [agent turn attempt {attempt+1}] {e} — retrying in {wait}s")
                time.sleep(wait)

        if agent_text is None:
            print("    ✗ Agent turn failed after 3 retries — ending conversation early.")
            break

        # Strip any accidental "Agent:" prefix
        agent_text = re.sub(r"^Agent:\s*", "", agent_text, flags=re.IGNORECASE).strip()

        history.append({"speaker": "Agent", "content": agent_text})
        messages.append({"role": "assistant", "content": agent_text})

        # Small rate-limit pause between full turn-pairs
        if args.provider == "anthropic":
            time.sleep(0.4)

    # ── Detect whether user pushed back on the planted error (Type B) ──
    if mode == "type_b" and messages:
        pushback_phrases = [
            "that's not right", "that's incorrect", "are you sure", "i thought",
            "i was told", "i've read", "according to", "that doesn't sound right",
            "wait,", "hold on", "actually,", "i don't think that's", "you said earlier",
            "but earlier", "you just said", "that contradicts", "that can't be right",
            "that's wrong", "no, it's", "i believe it's", "reg e", "regulation e",
        ]
        # Only check user turns after the first 2 pairs (give the error time to appear)
        late_user_turns = [
            m["content"].lower()
            for m in messages[4:]
            if m["role"] == "user"
        ]
        user_caught_error = any(
            any(phrase in turn for phrase in pushback_phrases)
            for turn in late_user_turns
        )

    # Build probe metadata
    probe_meta: dict = {
        "probing_type":   "inadvertent" if mode == "type_a" else "adversarial",
        "goal_completed": goal_completed,
    }
    if mode == "type_a":
        probe_meta["wrong_prior_belief"] = prior_belief
    elif mode == "type_b" and planted_error is not None:
        probe_meta["agent_failure_mode"]  = planted_error.agent_failure_mode
        probe_meta["planted_error"]       = planted_error.description
        probe_meta["user_caught_error"]   = user_caught_error

    return messages, probe_meta


# ─────────────────────────────────────────────
# TRAINING FORMAT CONVERSION
# ─────────────────────────────────────────────

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
    "If your goal is complete or you have decided to stop, respond with <eos>. "
    "Otherwise, generate your next message."
)


def to_training_format(
    messages:   list[dict],
    persona:    Persona,
    scenario:   Scenario,
    global_idx: int,
    probe_meta: dict,
) -> list[dict]:
    """Convert a generated conversation into training JSONL entries."""
    intent = generate_intent_summary(scenario)
    entries: list[dict] = []
    history_lines: list[str] = []

    for i, msg in enumerate(messages):
        if msg["role"] == "user":
            is_last_user = not any(m["role"] == "user" for m in messages[i + 1:])
            output = "<eos>" if is_last_user else msg["content"] + "<eos>"

            if not history_lines:
                input_text = FIRST_TURN_TEMPLATE.format(intent=intent)
            else:
                input_text = COMPLETION_TEMPLATE.format(
                    intent=intent,
                    history="\n".join(history_lines),
                )

            entries.append({
                "input":  input_text,
                "output": output,
                "_meta": {
                    "synthetic":      True,
                    "generation_idx": global_idx,
                    "persona":        asdict(persona),
                    "scenario":       asdict(scenario),
                    **probe_meta,
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
    dump_dir:   Path,
    idx:        int,
    persona:    Persona,
    scenario:   Scenario,
    messages:   list[dict],
    probe_meta: dict,
) -> Path:
    dump_dir.mkdir(parents=True, exist_ok=True)
    out = dump_dir / f"conversation_{idx + 1:03d}.txt"

    lines = []
    lines.append("=" * 60)
    lines.append(f"CONVERSATION {idx + 1}")
    lines.append("=" * 60)
    lines.append("")
    lines.append("PERSONA:")
    lines.append(f"  knowledge_level     : {persona.knowledge_level}")
    lines.append(f"  emotional_state     : {persona.emotional_state}")
    lines.append(f"  communication_style : {persona.communication_style}")
    lines.append(f"  goal_clarity        : {persona.goal_clarity}")
    lines.append("")
    lines.append("SCENARIO:")
    lines.append(f"  certainty               : {scenario.certainty}")
    lines.append(f"  information_completeness: {scenario.information_completeness}")
    lines.append(f"  prior_contact           : {scenario.prior_contact}")
    lines.append(f"  expected_resolution     : {scenario.expected_resolution}")
    lines.append(f"  merchant                : {scenario.merchant_name}")
    lines.append(f"  amount                  : {scenario.amount}")
    lines.append(f"  transaction_date        : {scenario.transaction_date}")
    lines.append("")
    lines.append("PROBE META:")
    for k, v in probe_meta.items():
        lines.append(f"  {k}: {v}")
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

def generate_all(args) -> None:
    print(f"\n=== UserLM Probe Generator  [mode: {args.mode}] ===\n")

    # generation_idx offsets keep each mode in a distinct range:
    #   success  →   0 –  99,999  (generate_conversations.py)
    #   failure  → 100,000 – 199,999  (generate_conversations.py)
    #   type_a   → 200,000 – 299,999
    #   type_b   → 300,000 – 399,999
    idx_offset = 200_000 if args.mode == "type_a" else 300_000

    combinations = [
        (combo_idx, scenario_idx, persona, scenario)
        for combo_idx, (persona, scenario_idx, scenario) in enumerate(
            (p, si, s)
            for p in PERSONAS
            for si, s in enumerate(SCENARIOS)
        )
    ]
    total_available = len(combinations)

    limit = total_available if args.limit == -1 else min(args.limit, total_available)
    limit_label = "all" if args.limit == -1 else str(limit)

    print(f"Generating {limit_label} / {total_available} conversations "
          f"({len(PERSONAS)} personas × {len(SCENARIOS)} scenarios)")
    print(f"Provider : {args.provider}")
    print(f"Output   : {args.output}\n")

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump_dir = output_path.parent / f"conversation_dumps_{args.mode}"

    # Resume: skip already-written generation_idxs
    generated_indices: set[int] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    e = json.loads(line)
                    gidx = e.get("_meta", {}).get("generation_idx")
                    if gidx is not None:
                        generated_indices.add(gidx)
                except Exception:
                    pass
        if generated_indices:
            print(f"  Resuming — {len(generated_indices)} conversations already done.\n")

    success_count = 0
    error_count   = 0

    for combo_idx, scenario_idx, persona, scenario in combinations:
        if success_count >= limit:
            break

        global_idx = idx_offset + combo_idx
        if global_idx in generated_indices:
            print(f"  [{combo_idx + 1}/{total_available}] Skipping (already done)")
            continue

        # Sample content for this conversation (seeded for reproducibility)
        rng = random.Random(args.seed * 1_000_000 + global_idx)
        content = sample_content(rng)
        scenario = Scenario(
            certainty=scenario.certainty,
            information_completeness=scenario.information_completeness,
            prior_contact=scenario.prior_contact,
            expected_resolution=scenario.expected_resolution,
            merchant_name=content["merchant_name"],
            amount=content["amount"],
            transaction_date=content["transaction_date"],
            card_type=content["card_type"],
            channel=content["channel"],
        )

        # Sample prior belief, planted error, and target message length independently
        prior_belief   = rng.choice(PRIOR_BELIEFS_POOL)
        planted_error  = rng.choice(PLANTED_ERRORS_POOL) if args.mode == "type_b" else None
        target_length  = sample_target_length(rng, persona.communication_style)

        label = (
            f"[{combo_idx + 1}/{total_available}] "
            f"{persona.knowledge_level}/{persona.emotional_state} | "
            f"{scenario.merchant_name} {scenario.amount} | "
            f"scenario_{scenario_idx}"
        )
        print(label)

        # ── Dry run: print prompt sizes and planted error ──
        if args.provider == "dry_run":
            user_sys  = build_user_system_prompt(persona, scenario, args.mode, prior_belief, target_length)
            agent_sys = build_agent_system_prompt(args.mode, planted_error)
            print(f"  USER  system prompt : {len(user_sys):,} chars")
            print(f"  AGENT system prompt : {len(agent_sys):,} chars")
            if args.mode == "type_b" and planted_error:
                print(f"  Planted error       : [{planted_error.agent_failure_mode}] {planted_error.description[:80]}...")
            elif args.mode == "type_a":
                print(f"  Prior belief        : {prior_belief}")
            success_count += 1
            continue

        # ── Real generation ──
        messages, probe_meta = generate_conversation_alternating(
            persona, scenario, args.mode, args,
            prior_belief=prior_belief, planted_error=planted_error,
            target_length=target_length,
        )

        if len(messages) < 2:
            print(f"  ✗ Conversation too short ({len(messages)} turns) — skipping.")
            error_count += 1
            continue

        training_entries = to_training_format(
            messages, persona, scenario, global_idx, probe_meta
        )

        with open(output_path, "a") as f:
            for entry in training_entries:
                f.write(json.dumps(entry) + "\n")

        dump_path = write_conversation_dump(
            dump_dir, combo_idx, persona, scenario, messages, probe_meta
        )

        user_turns = sum(1 for m in messages if m["role"] == "user")
        extra = ""
        if args.mode == "type_b":
            extra = f" | caught_error={probe_meta.get('user_caught_error')}"
        elif args.mode == "type_a":
            extra = f" | prior_belief_injected=True"

        print(
            f"  ✓ {len(messages)} turns ({user_turns} user) | "
            f"goal_complete={probe_meta['goal_completed']}{extra} "
            f"→ {len(training_entries)} training examples"
        )
        print(f"    Dump: {dump_path}")
        success_count += 1

        time.sleep(0.5)

    # ── Summary ──
    print(f"\n{'=' * 40}")
    if args.provider == "dry_run":
        print(f"Dry run complete. {success_count} prompt pairs previewed.")
        print("Re-run with --provider anthropic to generate conversations.")
    else:
        print(f"Done. {success_count} generated, {error_count} failed.")

        # Quick stats
        try:
            total_entries = 0
            eos_count     = 0
            caught_count  = 0
            with open(output_path) as f:
                for line in f:
                    e = json.loads(line)
                    total_entries += 1
                    if e.get("output") == "<eos>":
                        eos_count += 1
                    if e.get("_meta", {}).get("user_caught_error") is True:
                        caught_count += 1
            print(f"\nTotal training entries : {total_entries}")
            print(f"  Complete convos (<eos>)  : {eos_count}")
            if args.mode == "type_b":
                print(f"  Error caught by user     : {caught_count} entries")
            print(f"\nRun split_dataset.py to combine and split all datasets into train/val.")
        except Exception:
            pass


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Type A / Type B probe conversations for UserLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["type_a", "type_b"], required=True,
        help=(
            "type_a: inadvertent probing (info drip + prior beliefs + error awareness). "
            "type_b: adversarial probing (planted agent error + user pushback)."
        ),
    )
    parser.add_argument(
        "--provider", choices=["anthropic", "dry_run"], required=True,
        help="anthropic: call the API. dry_run: print prompts only.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSONL path. Defaults to probe_conversations_<mode>.jsonl",
    )
    parser.add_argument(
        "--limit", default=3,
        help="Conversations to generate. Default 3 (test). Pass 'all' for full set.",
    )
    parser.add_argument(
        "--api-key", default=os.environ.get("ANTHROPIC_API_KEY"),
        help="Anthropic API key. Defaults to ANTHROPIC_API_KEY env var.",
    )
    parser.add_argument(
        "--model", default="claude-sonnet-4-6",
        help="Anthropic model name (default: claude-sonnet-4-6).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducible content sampling (default: 42).",
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = f"probe_conversations_{args.mode}.jsonl"

    if str(args.limit).lower() == "all":
        args.limit = -1
    else:
        try:
            args.limit = int(args.limit)
            if args.limit < 1 and args.limit != -1:
                parser.error("--limit must be a positive integer or 'all'.")
        except ValueError:
            parser.error(f"--limit: invalid value '{args.limit}'.")

    if args.provider == "anthropic" and not args.api_key:
        parser.error("--api-key required (or set ANTHROPIC_API_KEY env var).")

    generate_all(args)


if __name__ == "__main__":
    main()
