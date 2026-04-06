"""
eval_conversations.py
Layer 1 statistical evaluation of synthetic banking dispute conversations.

Measures 5 dimensions — all rule-based, no API calls needed:
  A. Structural stats         — turn lengths, turn counts, length variance
  B. Lexical signals          — hedging rate vs over-certainty rate (Wang et al.)
  C. First-turn info density  — how much of {merchant, amount, date} the customer
                                volunteers in turn 1 vs. spreads across turns
  D. Breadth-first bundling   — how many concern-categories co-occur per turn
                                (proxy for LLM-like breadth-first questioning)
  E. Persona consistency      — emotional state markers, jargon vs knowledge level,
                                turn length vs communication style

Outputs:
  - eval_results.csv          — one row per conversation, all scores
  - images/01–07_*.png        — professional visualisation suite
  - Terminal summary          — aggregate stats + flagged outlier conversations
"""

import re
import csv
import json
import math
from pathlib import Path
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from typing import Optional

import matplotlib
matplotlib.use("Agg")                   # non-interactive backend — safe on all systems
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

DUMPS_DIR   = Path(__file__).parent / "conversation_dumps"
OUTPUT_CSV  = Path(__file__).parent / "eval_results.csv"
IMAGES_DIR  = Path(__file__).parent / "images"

# Flag a conversation if it hits >= this many dimension warnings
OUTLIER_THRESHOLD = 3

# ── Shared visual style ────────────────────────────────────────────────────────
PALETTE = {
    # emotional states
    "calm":              "#4C9BE8",
    "mildly_frustrated": "#F4A261",
    "very_upset":        "#E76F51",
    "escalating":        "#C1121F",
    # communication styles
    "direct":            "#2A9D8F",
    "indirect":          "#E9C46A",
    # knowledge levels
    "novice":            "#BDE0FE",
    "intermediate":      "#6A8EAE",
    "expert":            "#1B3A4B",
    # warning categories
    "structural":        "#457B9D",
    "lexical":           "#E9C46A",
    "density":           "#2A9D8F",
    "breadth":           "#E76F51",
    "persona":           "#A8DADC",
}

def _set_style():
    """Apply consistent seaborn/matplotlib theme."""
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "font.family":      "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.titlesize":   13,
        "axes.labelsize":   11,
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
    })

# ─────────────────────────────────────────────
#  Lexical word / phrase lists  (all lowercase)
# ─────────────────────────────────────────────

HEDGE_PHRASES = [
    "not sure", "i'm not sure", "im not sure", "i think", "i believe",
    "maybe", "perhaps", "possibly", "probably", "might", "could be",
    "i guess", "i don't know", "i dont know", "idk", "honestly",
    "sort of", "kind of", "not certain", "not 100%", "not totally sure",
    "i'm not certain", "im not certain",
]

# LLM-generated users over-use these (Wang et al. 2025 — "promise language")
CERTAINTY_PHRASES = [
    "definitely", "absolutely", "certainly", "for sure", "100%",
    "no question", "without a doubt", "clearly", "obviously", "undoubtedly",
    "no doubt",
]

# Mild frustration markers — expected for emotional_state: mildly_frustrated
MILD_FRUSTRATION_MARKERS = [
    "look", "already said", "already told", "already mentioned",
    "already explained", "come on", "seriously", "just want",
    "all i want", "that's why", "thats why",
]

# Strong frustration / escalation markers — expected for emotional_state: escalating / very_upset
STRONG_FRUSTRATION_MARKERS = [
    "unacceptable", "ridiculous", "outrageous", "i can't believe",
    "i cannot believe", "how is this possible", "this is insane",
    "close my account", "closing my account", "complaint", "cfpb",
    "lawyer", "attorney", "legal", "sue", "escalate", "manager",
    "supervisor", "not acceptable",
]

# Specialist banking jargon — novices should not initiate these terms
# We flag if they appear in a customer turn BEFORE the agent has used the term
BANKING_JARGON = [
    "provisional credit", "chargeback", "reg e", "regulation e",
    "billing error", "fraud claim", "dispute resolution",
    "zero liability", "fcba",
]

# ── Dropout language — per failure_mode (Layer 1, Section F) ──────────────────
# Each list contains surface signals we expect in the final customer turn when
# the dropout is genuine.  If none are found, DROPOUT_NOT_PLAUSIBLE is raised.
DROPOUT_SIGNALS: dict[str, list[str]] = {
    "impatience": [
        "forget it", "never mind", "this is taking", "too long", "too much time",
        "give up", "done with this", "can't do this", "bye", "goodbye", "later",
        "not worth", "waste of time",
    ],
    "info_blocker": [
        "come back", "come back later", "when i find", "when i have", "when i get",
        "get back to you", "try again later", "call back", "once i have",
        "find my", "locate my", "check my",
    ],
    "loop_exit": [
        "same question", "keep asking", "already told you", "already said",
        "asking again", "going in circles", "round in circles", "over and over",
        "already answered", "already gave you",
    ],
    "trust_breakdown": [
        "don't trust", "not confident", "speak to someone else", "this isn't working",
        "unhelpful", "useless", "not helpful", "doesn't help", "waste", "no point",
        "pointless", "getting nowhere",
    ],
    "escalation_exit": [
        "supervisor", "manager", "speak to someone", "real person", "human",
        "escalate", "done here", "going elsewhere", "going to call",
        "speaking to someone else", "not dealing with this",
    ],
    "wrong_channel": [
        "phone", "branch", "in person", "call you", "come in", "in branch",
        "different channel", "another way", "call the number", "visit",
    ],
    "scope_limit": [
        "can't help", "can't do anything", "thanks anyway", "never mind then",
        "forget it", "i'll try elsewhere", "ok then", "nothing you can do",
        "your hands are tied", "nothing to be done",
    ],
    "distraction": [
        "gotta go", "got to go", "have to go", "brb", "talk later",
        "something came up", "busy", "back later", "catch you later",
        "deal with this later", "have to run",
    ],
}

# Pushback phrases — expected in customer turns when agent makes an error (Type B)
PUSHBACK_PHRASES = [
    "actually", "i don't think that's right", "i thought", "are you sure",
    "that doesn't sound right", "that's not what", "wait,", "hold on",
    "but i thought", "i was told", "i believe", "shouldn't it be",
    "is that correct", "can you double check", "that doesn't seem right",
    "pretty sure", "i read", "i heard", "i checked", "i looked it up",
    "not what i", "i don't think so", "that seems wrong",
]

# Concern categories for breadth-first detection
CONCERN_CATEGORIES = {
    "dispute": [
        "dispute", "didn't make", "did not make", "don't recognize",
        "do not recognize", "didn't authorize", "did not authorize",
        "unauthorized", "fraudulent", "fraud", "charge i", "charge on",
        "transaction i", "not mine", "never made",
    ],
    "card_action": [
        "cancel", "lock", "block", "replace", "new card",
        "card replaced", "shut it down", "close the card",
    ],
    "refund_credit": [
        "money back", "refund", "get it back", "reimburse",
        "provisional credit", "provisional", "credit back",
        "when do i get", "get my money",
    ],
    "timeline_status": [
        "how long", "when will", "how many days", "timeline",
        "status", "update", "what's the eta", "whats the eta",
        "estimated time",
    ],
}

# ─────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────

@dataclass
class Turn:
    speaker: str   # "Customer" or "Agent"
    text: str

@dataclass
class Conversation:
    idx: int
    # Persona fields
    knowledge_level: str     = ""   # novice / intermediate / expert
    emotional_state: str     = ""   # calm / mildly_frustrated / very_upset / escalating
    communication_style: str = ""   # direct / indirect
    goal_clarity: str        = ""   # clear / vague / wrong_mental_model
    # Scenario fields
    certainty: str              = ""   # uncertain / certain_fraud
    information_completeness: str = "" # full / partial
    prior_contact: str           = ""   # first_contact / tried_merchant
    expected_resolution: str     = ""   # investigation / refund_and_card / ...
    merchant: str                = ""
    amount: str                  = ""
    transaction_date: str        = ""
    # Dialogue
    turns: list = field(default_factory=list)
    # EXTRA META — populated from dump file header; used for type-specific checks
    dataset_type: str        = "success"  # success | failure | type_a | type_b
    goal_completed: bool     = True
    failure_mode: str        = ""         # impatience | info_blocker | ... (failure only)
    probing_type: str        = ""         # inadvertent | adversarial (probe only)
    wrong_prior_belief: str  = ""         # type_a only
    agent_failure_mode: str  = ""         # type_b only
    planted_error: str       = ""         # type_b only
    user_caught_error: Optional[bool] = None  # type_b only

@dataclass
class EvalResult:
    conv_idx: int
    dataset_type: str
    knowledge_level: str
    emotional_state: str
    communication_style: str
    goal_clarity: str
    certainty: str
    # A. Structural
    num_customer_turns: int    = 0
    words_mean: float          = 0.0
    words_median: float        = 0.0
    words_p10: float           = 0.0
    words_p90: float           = 0.0
    words_stddev: float        = 0.0
    # B. Lexical
    hedge_rate: float          = 0.0   # hedging phrases per customer turn
    certainty_rate: float      = 0.0   # certainty phrases per customer turn
    hedge_minus_certainty: float = 0.0 # positive = more hedging (human-like)
    # C. Info density
    facts_in_turn1: int        = 0     # out of 3 (merchant, amount, date)
    info_density_score: float  = 0.0   # facts_in_turn1 / 3
    # D. Breadth-first bundling
    max_categories_per_turn: int   = 0  # worst single turn
    turns_with_multi_category: int = 0  # turns bundling 2+ categories
    breadth_first_flag: bool       = False
    # E. Persona consistency
    frustration_marker_present: bool = False  # any mild/strong marker found
    jargon_before_agent: bool        = False  # novice used jargon first
    # Warnings
    warnings: list = field(default_factory=list)
    warning_count: int = 0
    is_outlier: bool   = False

# ─────────────────────────────────────────────
#  Parser
# ─────────────────────────────────────────────

def parse_conversation(filepath: Path) -> Optional[Conversation]:
    text = filepath.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Extract conversation index from header
    idx = None
    for line in lines:
        m = re.match(r"CONVERSATION\s+(\d+)", line.strip())
        if m:
            idx = int(m.group(1))
            break
    if idx is None:
        return None

    conv = Conversation(idx=idx)

    # ── Parse PERSONA, SCENARIO, EXTRA META, and dialogue ──
    in_persona   = False
    in_scenario  = False
    in_extra_meta = False
    in_dialogue  = False

    for line in lines:
        stripped = line.strip()

        if stripped == "PERSONA:":
            in_persona = True;  in_scenario = False; in_extra_meta = False; continue
        if stripped == "SCENARIO:":
            in_scenario = True; in_persona  = False; in_extra_meta = False; continue
        if stripped == "EXTRA META:":
            in_extra_meta = True; in_persona = False; in_scenario = False; continue
        if re.match(r"-{4,}", stripped):
            in_persona = False; in_scenario = False; in_extra_meta = False; continue

        # Key : value pairs (indented with spaces, colon separator)
        kv = re.match(r"\s+([\w_]+)\s*:\s*(.+)", line)
        if kv:
            key   = kv.group(1).strip().lower()
            value = kv.group(2).strip()
            if in_persona:
                if key == "knowledge_level":       conv.knowledge_level       = value
                elif key == "emotional_state":     conv.emotional_state       = value
                elif key == "communication_style": conv.communication_style   = value
                elif key == "goal_clarity":        conv.goal_clarity          = value
            elif in_scenario:
                if key == "certainty":                  conv.certainty                  = value
                elif key == "information_completeness": conv.information_completeness   = value
                elif key == "prior_contact":            conv.prior_contact              = value
                elif key == "expected_resolution":      conv.expected_resolution        = value
                elif key == "merchant":                 conv.merchant                   = value
                elif key == "amount":                   conv.amount                     = value
                elif key == "transaction_date":         conv.transaction_date           = value
            elif in_extra_meta:
                if key == "goal_completed":
                    conv.goal_completed = value.lower() not in ("false", "0", "no")
                elif key == "failure_mode":
                    conv.failure_mode = value
                elif key == "probing_type":
                    conv.probing_type = value
                elif key == "wrong_prior_belief":
                    conv.wrong_prior_belief = value
                elif key == "agent_failure_mode":
                    conv.agent_failure_mode = value
                elif key == "planted_error":
                    conv.planted_error = value
                elif key == "user_caught_error":
                    conv.user_caught_error = value.lower() not in ("false", "0", "no")
            continue

        # Dialogue turns: "Customer: ..." or "Agent: ..."
        m = re.match(r"^(Customer|Agent):\s*(.+)", stripped)
        if m:
            in_dialogue   = True
            in_persona    = False
            in_scenario   = False
            in_extra_meta = False
            conv.turns.append(Turn(speaker=m.group(1), text=m.group(2).strip()))
            continue

        # Multi-line continuation within a turn
        if in_dialogue and conv.turns and stripped and not re.match(r"=+|-{4,}", stripped):
            if not re.match(r"^(Customer|Agent):", stripped):
                conv.turns[-1].text += " " + stripped

    # Derive dataset_type from parsed meta
    if conv.probing_type == "inadvertent":
        conv.dataset_type = "type_a"
    elif conv.probing_type == "adversarial":
        conv.dataset_type = "type_b"
    elif not conv.goal_completed:
        conv.dataset_type = "failure"
    else:
        conv.dataset_type = "success"

    return conv

# ─────────────────────────────────────────────
#  Helper utilities
# ─────────────────────────────────────────────

def word_count(text: str) -> int:
    return len(text.split())

def percentile(data: list, p: float) -> float:
    if not data:
        return 0.0
    data_sorted = sorted(data)
    idx = (len(data_sorted) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(data_sorted) - 1)
    return data_sorted[lo] + (data_sorted[hi] - data_sorted[lo]) * (idx - lo)

def stddev(data: list) -> float:
    if len(data) < 2:
        return 0.0
    mean = sum(data) / len(data)
    return math.sqrt(sum((x - mean) ** 2 for x in data) / len(data))

def contains_phrase(text_lower: str, phrases: list) -> list:
    """Return list of matching phrases found in text (case-insensitive, already lowered)."""
    return [p for p in phrases if p in text_lower]

def count_phrases(text_lower: str, phrases: list) -> int:
    return sum(1 for p in phrases if p in text_lower)

def extract_amount_digits(amount_str: str) -> str:
    """'$1,249.00' → '1249.00' for fuzzy matching."""
    return re.sub(r"[^\d.]", "", amount_str)

def extract_date_day(date_str: str) -> str:
    """'August 14th' → '14'  for checking if the day number appears."""
    m = re.search(r"(\d+)", date_str)
    return m.group(1) if m else ""

def extract_date_month(date_str: str) -> str:
    """'August 14th' → 'august'."""
    m = re.match(r"([a-zA-Z]+)", date_str.strip())
    return m.group(1).lower() if m else ""

# ─────────────────────────────────────────────
#  Core evaluation logic
# ─────────────────────────────────────────────

def evaluate(conv: Conversation) -> EvalResult:
    result = EvalResult(
        conv_idx            = conv.idx,
        dataset_type        = conv.dataset_type,
        knowledge_level     = conv.knowledge_level,
        emotional_state     = conv.emotional_state,
        communication_style = conv.communication_style,
        goal_clarity        = conv.goal_clarity,
        certainty           = conv.certainty,
    )
    warnings = []

    customer_turns = [t for t in conv.turns if t.speaker == "Customer"]
    agent_turns    = [t for t in conv.turns if t.speaker == "Agent"]

    if not customer_turns:
        warnings.append("NO_CUSTOMER_TURNS")
        result.warnings = warnings
        result.warning_count = len(warnings)
        return result

    # ── A. Structural stats ──────────────────────────────────────────────
    wc_list = [word_count(t.text) for t in customer_turns]
    result.num_customer_turns = len(customer_turns)
    result.words_mean   = round(sum(wc_list) / len(wc_list), 1)
    result.words_median = round(percentile(wc_list, 50), 1)
    result.words_p10    = round(percentile(wc_list, 10), 1)
    result.words_p90    = round(percentile(wc_list, 90), 1)
    result.words_stddev = round(stddev(wc_list), 1)

    # Flag very long average turns — LLMs tend to over-explain
    if result.words_mean > 30:
        warnings.append(f"HIGH_AVG_TURN_LENGTH ({result.words_mean:.1f} words/turn)")

    # Flag very low stddev — unnaturally uniform turn lengths
    if result.words_stddev < 3.0 and len(wc_list) >= 3:
        warnings.append(f"LOW_LENGTH_VARIANCE (stddev={result.words_stddev})")

    # ── B. Lexical signals ───────────────────────────────────────────────
    total_hedge     = 0
    total_certainty = 0

    for turn in customer_turns:
        tl = turn.text.lower()
        total_hedge     += count_phrases(tl, HEDGE_PHRASES)
        total_certainty += count_phrases(tl, CERTAINTY_PHRASES)

    n = len(customer_turns)
    result.hedge_rate          = round(total_hedge     / n, 3)
    result.certainty_rate      = round(total_certainty / n, 3)
    result.hedge_minus_certainty = round(result.hedge_rate - result.certainty_rate, 3)

    # Flag: uncertain/mildly_frustrated personas should hedge more than they commit
    uncertain_persona = conv.emotional_state in ("calm", "mildly_frustrated") \
                     or conv.certainty == "uncertain"
    if uncertain_persona and result.certainty_rate > result.hedge_rate:
        warnings.append(
            f"CERTAINTY_EXCEEDS_HEDGING "
            f"(certainty={result.certainty_rate:.2f}, hedge={result.hedge_rate:.2f})"
        )

    # Flag: zero hedging across the whole conversation is suspicious for uncertain personas
    if conv.certainty == "uncertain" and total_hedge == 0:
        warnings.append("ZERO_HEDGING_FOR_UNCERTAIN_SCENARIO")

    # ── C. First-turn info density ───────────────────────────────────────
    turn1_text = customer_turns[0].text.lower()

    merchant_in_t1 = conv.merchant.lower() in turn1_text if conv.merchant else False

    # Amount: check if the raw digits appear (handles "$1,249.00" → "1249")
    amount_digits = extract_amount_digits(conv.amount)
    amount_in_t1  = bool(amount_digits and amount_digits in turn1_text.replace(",", ""))

    # Date: check if day number AND/OR month name appear
    day   = extract_date_day(conv.transaction_date)
    month = extract_date_month(conv.transaction_date)
    date_in_t1 = bool(
        (day   and day   in turn1_text) or
        (month and month in turn1_text)
    )

    facts_present = sum([merchant_in_t1, amount_in_t1, date_in_t1])
    result.facts_in_turn1    = facts_present
    result.info_density_score = round(facts_present / 3, 3)

    # Flag: dumping all 3 facts in turn 1 is LLM-like (real users rarely do this)
    if facts_present == 3:
        warnings.append(
            "INFO_DUMP_TURN1 — all 3 facts (merchant, amount, date) in first customer turn"
        )

    # ── D. Breadth-first bundling ────────────────────────────────────────
    max_cats   = 0
    multi_turns = 0

    for turn in customer_turns:
        tl = turn.text.lower()
        cats_hit = []
        for cat_name, phrases in CONCERN_CATEGORIES.items():
            if any(p in tl for p in phrases):
                cats_hit.append(cat_name)
        n_cats = len(cats_hit)
        if n_cats > max_cats:
            max_cats = n_cats
        if n_cats >= 2:
            multi_turns += 1

    result.max_categories_per_turn   = max_cats
    result.turns_with_multi_category = multi_turns
    result.breadth_first_flag        = max_cats >= 3 or multi_turns >= 2

    if result.breadth_first_flag:
        warnings.append(
            f"BREADTH_FIRST_BUNDLING "
            f"(max_cats={max_cats}, multi_cat_turns={multi_turns})"
        )

    # ── E. Persona consistency ───────────────────────────────────────────

    # E1. Emotional state — frustration markers
    all_customer_text = " ".join(t.text.lower() for t in customer_turns)

    mild_hits   = contains_phrase(all_customer_text, MILD_FRUSTRATION_MARKERS)
    strong_hits = contains_phrase(all_customer_text, STRONG_FRUSTRATION_MARKERS)

    result.frustration_marker_present = bool(mild_hits or strong_hits)

    state = conv.emotional_state
    if state in ("mildly_frustrated",) and not mild_hits and not strong_hits:
        warnings.append(
            "MISSING_FRUSTRATION_MARKERS — persona is mildly_frustrated but no markers found"
        )
    if state == "calm" and (mild_hits or strong_hits):
        warnings.append(
            f"UNEXPECTED_FRUSTRATION — persona is calm but found markers: "
            f"{(mild_hits + strong_hits)[:3]}"
        )
    if state in ("very_upset", "escalating") and not strong_hits:
        warnings.append(
            f"MISSING_STRONG_FRUSTRATION — persona is {state} but no strong markers found"
        )

    # E2. Knowledge level — jargon before agent introduces it
    # Walk turn by turn; track which jargon the agent has said so far
    agent_said_jargon: set = set()
    jargon_violations: list = []

    turn_pairs = conv.turns  # interleaved Customer / Agent turns

    for i, turn in enumerate(turn_pairs):
        tl = turn.text.lower()
        if turn.speaker == "Agent":
            for term in BANKING_JARGON:
                if term in tl:
                    agent_said_jargon.add(term)
        elif turn.speaker == "Customer":
            for term in BANKING_JARGON:
                if term in tl and term not in agent_said_jargon:
                    jargon_violations.append(term)

    result.jargon_before_agent = bool(jargon_violations)

    if conv.knowledge_level == "novice" and jargon_violations:
        warnings.append(
            f"NOVICE_JARGON — novice customer used specialist term(s) before agent: "
            f"{jargon_violations}"
        )

    # E3. Communication style — direct personas should have shorter turns
    # (We collect this for population-level analysis; flag extreme mismatches here)
    if conv.communication_style == "direct" and result.words_mean > 25:
        warnings.append(
            f"DIRECT_STYLE_BUT_LONG_TURNS — avg {result.words_mean:.1f} words; "
            f"direct personas should be concise"
        )
    if conv.communication_style == "indirect" and result.words_mean < 6:
        warnings.append(
            f"INDIRECT_STYLE_BUT_VERY_SHORT — avg {result.words_mean:.1f} words"
        )

    # ── F. Type-specific checks ──────────────────────────────────────────
    dtype = conv.dataset_type

    if dtype == "failure":
        # F1a. Premature dropout — real dropout needs frustration build-up
        if len(customer_turns) < 3:
            warnings.append(
                f"DROPOUT_TOO_EARLY — only {len(customer_turns)} customer turn(s) before <eos>"
            )

        # F1b. Plausibility — final customer turn should carry mode-appropriate exit language
        if customer_turns:
            final_text = customer_turns[-1].text.lower()
            mode_signals = DROPOUT_SIGNALS.get(conv.failure_mode, [])
            if mode_signals and not any(sig in final_text for sig in mode_signals):
                warnings.append(
                    f"DROPOUT_NOT_PLAUSIBLE — no [{conv.failure_mode}] exit language "
                    f"in final customer turn"
                )

        # F1c. Escalation arc — impatience/escalation_exit must show tone deterioration
        if conv.failure_mode in ("impatience", "escalation_exit") and len(customer_turns) >= 4:
            half = len(customer_turns) // 2
            late_turns  = customer_turns[half:]
            late_text   = " ".join(t.text.lower() for t in late_turns)
            late_frustration = (
                count_phrases(late_text, MILD_FRUSTRATION_MARKERS) > 0 or
                count_phrases(late_text, STRONG_FRUSTRATION_MARKERS) > 0
            )
            if not late_frustration:
                warnings.append(
                    f"MISSING_ESCALATION_ARC — [{conv.failure_mode}] but no frustration "
                    f"markers in the second half of the conversation"
                )

    elif dtype == "type_a" and conv.wrong_prior_belief:
        # F2a. Prior belief must surface somewhere in the customer turns
        STOPWORDS = {
            "you", "your", "that", "this", "the", "and", "are", "for",
            "from", "with", "have", "will", "can", "not", "any", "all",
            "they", "their", "when", "what", "just", "only",
        }
        belief_kws = [
            w for w in re.findall(r"[a-z]+", conv.wrong_prior_belief.lower())
            if len(w) > 3 and w not in STOPWORDS
        ]
        all_cust_lower = " ".join(t.text.lower() for t in customer_turns)
        matched_kws    = [kw for kw in belief_kws if kw in all_cust_lower]

        if len(matched_kws) < 2:
            warnings.append(
                f"PRIOR_BELIEF_ABSENT — wrong prior belief keywords not found in "
                f"customer turns (looked for: {belief_kws[:6]})"
            )
        elif len(customer_turns) >= 4:
            # F2b. Belief must not vanish silently after appearing
            half       = len(customer_turns) // 2
            first_half = " ".join(t.text.lower() for t in customer_turns[:half])
            second_half= " ".join(t.text.lower() for t in customer_turns[half:])
            first_hits = sum(1 for kw in belief_kws if kw in first_half)
            second_hits= sum(1 for kw in belief_kws if kw in second_half)
            if first_hits >= 2 and second_hits == 0:
                warnings.append(
                    "BELIEF_SILENTLY_DROPPED — prior belief keywords appear in "
                    "first half but disappear in second half with no explicit resolution"
                )

    elif dtype == "type_b":
        all_cust_lower = " ".join(t.text.lower() for t in customer_turns)
        pushback_found = any(p in all_cust_lower for p in PUSHBACK_PHRASES)

        # F3a. Error not caught — no pushback language anywhere
        if not pushback_found:
            warnings.append(
                "ERROR_NOT_CAUGHT — no pushback phrases found; customer appears to "
                "accept the planted agent error without challenge"
            )

        # F3b. Label consistency — _meta user_caught_error vs actual transcript
        if conv.user_caught_error is True and not pushback_found:
            warnings.append(
                "USER_CAUGHT_ERROR_LABEL_MISMATCH — _meta says user_caught_error=True "
                "but no pushback language found in transcript"
            )
        elif conv.user_caught_error is False and pushback_found:
            warnings.append(
                "USER_CAUGHT_ERROR_LABEL_MISMATCH — _meta says user_caught_error=False "
                "but pushback language is present in transcript"
            )

    # ── Wrap up ──────────────────────────────────────────────────────────
    result.warnings      = warnings
    result.warning_count = len(warnings)
    result.is_outlier    = result.warning_count >= OUTLIER_THRESHOLD

    return result

# ─────────────────────────────────────────────
#  Population-level checks (run after all convs)
# ─────────────────────────────────────────────

def population_checks(results: list) -> list:
    """
    E3 population check: across all conversations, do 'direct' personas actually
    have shorter avg turn length than 'indirect' ones?  If not, flag all direct convs.
    """
    direct_means   = [r.words_mean for r in results if r.communication_style == "direct"]
    indirect_means = [r.words_mean for r in results if r.communication_style == "indirect"]

    population_notes = []
    if direct_means and indirect_means:
        direct_avg   = sum(direct_means)   / len(direct_means)
        indirect_avg = sum(indirect_means) / len(indirect_means)
        if direct_avg >= indirect_avg:
            note = (
                f"POPULATION_STYLE_MISMATCH — 'direct' avg turn length ({direct_avg:.1f}) "
                f">= 'indirect' avg ({indirect_avg:.1f}). "
                f"Communication style is not differentiating turn length."
            )
            population_notes.append(note)
        else:
            population_notes.append(
                f"POPULATION_STYLE_OK — direct={direct_avg:.1f} words/turn < "
                f"indirect={indirect_avg:.1f} words/turn ✓"
            )

    return population_notes

# ─────────────────────────────────────────────
#  CSV output
# ─────────────────────────────────────────────

def write_csv(results: list, path: Path):
    if not results:
        return
    fieldnames = [
        "conv_idx", "dataset_type", "knowledge_level", "emotional_state",
        "communication_style", "goal_clarity", "certainty",
        "num_customer_turns",
        "words_mean", "words_median", "words_p10", "words_p90", "words_stddev",
        "hedge_rate", "certainty_rate", "hedge_minus_certainty",
        "facts_in_turn1", "info_density_score",
        "max_categories_per_turn", "turns_with_multi_category", "breadth_first_flag",
        "frustration_marker_present", "jargon_before_agent",
        "warning_count", "is_outlier", "warnings",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = asdict(r)
            row["warnings"] = " | ".join(r.warnings)
            writer.writerow({k: row[k] for k in fieldnames})
    print(f"\n✓  Results written to: {path}")

# ─────────────────────────────────────────────
#  Terminal summary
# ─────────────────────────────────────────────

def print_summary(results: list, pop_notes: list):
    n = len(results)
    if n == 0:
        print("No conversations evaluated.")
        return

    sep = "─" * 70

    print(f"\n{'=' * 70}")
    print(f"  LAYER 1 EVALUATION SUMMARY  ({n} conversations)")
    print(f"{'=' * 70}")

    # ── A. Structural ───────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  A. STRUCTURAL STATS (customer turns only)")
    print(sep)
    all_means   = [r.words_mean          for r in results]
    all_medians = [r.words_median        for r in results]
    all_p90     = [r.words_p90           for r in results]
    all_stddev  = [r.words_stddev        for r in results]
    all_nturn   = [r.num_customer_turns  for r in results]

    def ag(lst): return f"mean={sum(lst)/len(lst):.1f}  min={min(lst):.1f}  max={max(lst):.1f}"

    print(f"  Avg words/turn (per conv mean):  {ag(all_means)}")
    print(f"  Median words/turn (per conv):    {ag(all_medians)}")
    print(f"  P90 words/turn (per conv):       {ag(all_p90)}")
    print(f"  Turn length stddev (per conv):   {ag(all_stddev)}")
    print(f"  # customer turns per conv:       {ag(all_nturn)}")

    hi_avg    = [r.conv_idx for r in results if r.words_mean > 30]
    lo_stddev = [r.conv_idx for r in results if r.words_stddev < 3.0 and r.num_customer_turns >= 3]
    if hi_avg:    print(f"\n  ⚠  HIGH_AVG_TURN_LENGTH (>30 words): convs {sorted(hi_avg)}")
    if lo_stddev: print(f"  ⚠  LOW_VARIANCE (<3.0 stddev):        convs {sorted(lo_stddev)}")

    # ── B. Lexical ──────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  B. LEXICAL SIGNALS — hedging vs over-certainty")
    print(sep)
    hedge_rates = [r.hedge_rate     for r in results]
    cert_rates  = [r.certainty_rate for r in results]
    hmc         = [r.hedge_minus_certainty for r in results]

    print(f"  Hedge rate     (phrases/turn): {ag(hedge_rates)}")
    print(f"  Certainty rate (phrases/turn): {ag(cert_rates)}")
    print(f"  Hedge − Certainty:             {ag(hmc)}")

    cert_exceed = [r.conv_idx for r in results
                   if "CERTAINTY_EXCEEDS_HEDGING" in " ".join(r.warnings)]
    zero_hedge  = [r.conv_idx for r in results
                   if "ZERO_HEDGING_FOR_UNCERTAIN_SCENARIO" in " ".join(r.warnings)]
    if cert_exceed: print(f"\n  ⚠  CERTAINTY > HEDGING (uncertain persona): convs {sorted(cert_exceed)}")
    if zero_hedge:  print(f"  ⚠  ZERO HEDGING (uncertain scenario):        convs {sorted(zero_hedge)}")

    # ── C. Info density ─────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  C. FIRST-TURN INFO DENSITY")
    print(sep)
    density_scores = [r.info_density_score for r in results]
    facts_dist     = defaultdict(int)
    for r in results:
        facts_dist[r.facts_in_turn1] += 1

    print(f"  Avg info density (0–1):  {sum(density_scores)/len(density_scores):.2f}")
    print(f"  Facts-in-turn-1 distribution:")
    for k in sorted(facts_dist):
        label = ["(none)", "(1 fact)", "(2 facts)", "(all 3 — LLM-like!)"][min(k, 3)]
        bar   = "█" * facts_dist[k]
        print(f"    {k} facts {label}: {facts_dist[k]:3d} convs  {bar}")

    info_dumps = [r.conv_idx for r in results if r.facts_in_turn1 == 3]
    if info_dumps:
        print(f"\n  ⚠  INFO_DUMP_TURN1 (all 3 facts):  convs {sorted(info_dumps)}")

    # ── D. Breadth-first bundling ────────────────────────────────────────
    print(f"\n{sep}")
    print("  D. BREADTH-FIRST TOPIC BUNDLING")
    print(sep)
    bf_flagged = [r.conv_idx for r in results if r.breadth_first_flag]
    max_cats   = [r.max_categories_per_turn   for r in results]
    multi_turns= [r.turns_with_multi_category for r in results]

    print(f"  Max concern-categories in one turn:  {ag(max_cats)}")
    print(f"  Turns bundling 2+ categories:        {ag(multi_turns)}")
    print(f"  Breadth-first flagged:               {len(bf_flagged)}/{n} convs")
    if bf_flagged:
        print(f"  ⚠  Flagged convs: {sorted(bf_flagged)}")

    # ── E. Persona consistency ───────────────────────────────────────────
    print(f"\n{sep}")
    print("  E. PERSONA CONSISTENCY")
    print(sep)

    missing_frustration = [r.conv_idx for r in results
                           if "MISSING_FRUSTRATION_MARKERS" in " ".join(r.warnings)]
    unexpected_calm     = [r.conv_idx for r in results
                           if "UNEXPECTED_FRUSTRATION" in " ".join(r.warnings)]
    missing_strong      = [r.conv_idx for r in results
                           if "MISSING_STRONG_FRUSTRATION" in " ".join(r.warnings)]
    novice_jargon       = [r.conv_idx for r in results
                           if "NOVICE_JARGON" in " ".join(r.warnings)]
    style_mismatch      = [r.conv_idx for r in results
                           if "DIRECT_STYLE_BUT_LONG_TURNS" in " ".join(r.warnings)
                              or "INDIRECT_STYLE_BUT_VERY_SHORT" in " ".join(r.warnings)]

    def pct(lst): return f"{len(lst)}/{n} ({100*len(lst)/n:.0f}%)"

    print(f"  Missing frustration markers (mildly_frustrated): {pct(missing_frustration)}")
    if missing_frustration: print(f"    convs: {sorted(missing_frustration)}")

    print(f"  Unexpected frustration (calm persona):           {pct(unexpected_calm)}")
    if unexpected_calm: print(f"    convs: {sorted(unexpected_calm)}")

    print(f"  Missing strong frustration (very_upset/esc.):    {pct(missing_strong)}")
    if missing_strong:       print(f"    convs: {sorted(missing_strong)}")

    print(f"  Novice using jargon before agent:                {pct(novice_jargon)}")
    if novice_jargon:        print(f"    convs: {sorted(novice_jargon)}")

    print(f"  Communication style / turn-length mismatch:      {pct(style_mismatch)}")
    if style_mismatch:       print(f"    convs: {sorted(style_mismatch)}")

    # ── F. Type-specific check summary ──────────────────────────────────────
    type_specific_keys = [
        "DROPOUT_TOO_EARLY", "DROPOUT_NOT_PLAUSIBLE", "MISSING_ESCALATION_ARC",
        "PRIOR_BELIEF_ABSENT", "BELIEF_SILENTLY_DROPPED",
        "ERROR_NOT_CAUGHT", "USER_CAUGHT_ERROR_LABEL_MISMATCH",
    ]
    has_any_type_specific = any(
        any(k in " ".join(r.warnings) for k in type_specific_keys)
        for r in results
    )
    if has_any_type_specific:
        print(f"\n{sep}")
        print("  F. TYPE-SPECIFIC CHECKS")
        print(sep)
        for key in type_specific_keys:
            flagged = [r.conv_idx for r in results if any(key in w for w in r.warnings)]
            if flagged:
                print(f"  ⚠  {key}: {pct(flagged)}")
                print(f"    convs: {sorted(flagged)}")

    # Population-level notes
    if pop_notes:
        print(f"\n  Population notes:")
        for note in pop_notes:
            print(f"    → {note}")

    # ── Outlier summary ─────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  OUTLIER SUMMARY  (≥ {OUTLIER_THRESHOLD} warnings per conversation)")
    print(sep)
    outliers = sorted([r for r in results if r.is_outlier], key=lambda r: -r.warning_count)

    if not outliers:
        print(f"  ✓  No outliers found — all conversations have < {OUTLIER_THRESHOLD} warnings.")
    else:
        print(f"  {len(outliers)} outlier conversation(s):\n")
        for r in outliers:
            print(f"  Conv {r.conv_idx:03d}  [{r.warning_count} warnings]  "
                  f"({r.emotional_state}, {r.knowledge_level}, {r.communication_style})")
            for w in r.warnings:
                print(f"    • {w}")
            print()

    print(f"{'=' * 70}\n")

# ─────────────────────────────────────────────
#  Visualisations
# ─────────────────────────────────────────────

def _save(fig, name: str):
    IMAGES_DIR.mkdir(exist_ok=True)
    path = IMAGES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {path.name}")


# ── Figure 1 — Warning Summary ────────────────────────────────────────────────
def fig_warning_summary(results: list):
    """
    Horizontal bar chart of every warning type, grouped and colour-coded by
    evaluation dimension (A–E).  Sorted by frequency within each group.
    """
    _set_style()

    # Map each warning prefix to a display label and colour
    WARNING_META = {
        # Structural
        "HIGH_AVG_TURN_LENGTH":     ("A – High avg turn length",   PALETTE["structural"]),
        "LOW_LENGTH_VARIANCE":      ("A – Low turn-length variance", PALETTE["structural"]),
        # Lexical
        "CERTAINTY_EXCEEDS_HEDGING":        ("B – Certainty > Hedging (uncertain persona)",  PALETTE["lexical"]),
        "ZERO_HEDGING_FOR_UNCERTAIN_SCENARIO": ("B – Zero hedging in uncertain scenario", PALETTE["lexical"]),
        # Info density
        "INFO_DUMP_TURN1":          ("C – Info dump in turn 1 (all 3 facts)", PALETTE["density"]),
        # Breadth-first
        "BREADTH_FIRST_BUNDLING":   ("D – Breadth-first topic bundling",      PALETTE["breadth"]),
        # Persona
        "MISSING_FRUSTRATION_MARKERS": ("E – Missing frustration markers",    PALETTE["persona"]),
        "UNEXPECTED_FRUSTRATION":      ("E – Unexpected frustration (calm)",  PALETTE["persona"]),
        "MISSING_STRONG_FRUSTRATION":  ("E – Missing strong frustration",     PALETTE["persona"]),
        "NOVICE_JARGON":               ("E – Novice used jargon first",       PALETTE["persona"]),
        "DIRECT_STYLE_BUT_LONG_TURNS":   ("E – Direct style but long turns",       PALETTE["persona"]),
        "INDIRECT_STYLE_BUT_VERY_SHORT": ("E – Indirect style but very short",      PALETTE["persona"]),
        # Type-specific (F)
        "DROPOUT_TOO_EARLY":             ("F – Dropout too early (< 3 turns)",      "#9B5DE5"),
        "DROPOUT_NOT_PLAUSIBLE":         ("F – Dropout language absent",            "#9B5DE5"),
        "MISSING_ESCALATION_ARC":        ("F – Missing escalation arc",             "#9B5DE5"),
        "PRIOR_BELIEF_ABSENT":           ("F – Prior belief absent from transcript","#00BBF9"),
        "BELIEF_SILENTLY_DROPPED":       ("F – Prior belief silently dropped",      "#00BBF9"),
        "ERROR_NOT_CAUGHT":              ("F – Planted error not caught",           "#F15BB5"),
        "USER_CAUGHT_ERROR_LABEL_MISMATCH": ("F – user_caught_error label mismatch","#F15BB5"),
    }

    counts = defaultdict(int)
    for r in results:
        seen = set()
        for w in r.warnings:
            for key in WARNING_META:
                if w.startswith(key) and key not in seen:
                    counts[key] += 1
                    seen.add(key)

    # Build ordered lists
    ordered_keys = [k for k in WARNING_META if counts[k] > 0]
    ordered_keys.sort(key=lambda k: counts[k], reverse=True)

    labels = [WARNING_META[k][0] for k in ordered_keys]
    values = [counts[k]          for k in ordered_keys]
    colors = [WARNING_META[k][1] for k in ordered_keys]

    fig, ax = plt.subplots(figsize=(11, max(4, len(ordered_keys) * 0.55 + 1.2)))

    bars = ax.barh(range(len(ordered_keys)), values, color=colors, edgecolor="white",
                   linewidth=0.6, height=0.65)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", ha="left", fontsize=10, fontweight="bold")

    ax.set_yticks(range(len(ordered_keys)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Number of conversations flagged  (out of 80)")
    ax.set_title("Warning Frequency by Evaluation Dimension\n"
                 "How often each quality signal is triggered across 80 conversations",
                 pad=12)
    ax.set_xlim(0, max(values) * 1.18)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Legend patches for dimensions
    legend_items = [
        mpatches.Patch(color=PALETTE["structural"], label="A – Structural"),
        mpatches.Patch(color=PALETTE["lexical"],    label="B – Lexical signals"),
        mpatches.Patch(color=PALETTE["density"],    label="C – Info density"),
        mpatches.Patch(color=PALETTE["breadth"],    label="D – Breadth-first bundling"),
        mpatches.Patch(color=PALETTE["persona"],    label="E – Persona consistency"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=9,
              framealpha=0.9, edgecolor="#cccccc")

    fig.tight_layout()
    _save(fig, "01_warning_summary.png")


# ── Figure 2 — Structural Stats ───────────────────────────────────────────────
def fig_structural_stats(results: list):
    """
    2×2 grid covering all four structural metrics:
    avg words/turn, turn-length stddev, customer turn count, and p10/median/p90.
    """
    _set_style()
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("A – Structural Statistics of Customer Turns\n"
                 "Distribution of turn-length and conversation-length metrics across 80 conversations",
                 fontsize=14, fontweight="bold", y=1.01)

    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
    C = PALETTE["structural"]

    means   = [r.words_mean         for r in results]
    stddevs = [r.words_stddev       for r in results]
    nturns  = [r.num_customer_turns for r in results]
    p10s    = [r.words_p10          for r in results]
    p50s    = [r.words_median       for r in results]
    p90s    = [r.words_p90          for r in results]

    # ── 2a: histogram avg words/turn ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(means, bins=16, color=C, edgecolor="white", linewidth=0.7)
    ax1.axvline(30, color="#C1121F", linewidth=1.6, linestyle="--",
                label="Flag threshold (30 words)")
    ax1.axvline(np.mean(means), color="#1B3A4B", linewidth=1.4, linestyle=":",
                label=f"Dataset mean ({np.mean(means):.1f})")
    ax1.set_xlabel("Average words per turn")
    ax1.set_ylabel("Number of conversations")
    ax1.set_title("Avg Words per Customer Turn\n(per conversation)")
    ax1.legend(fontsize=9)

    # ── 2b: histogram turn-length stddev ──────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(stddevs, bins=16, color=C, edgecolor="white", linewidth=0.7)
    ax2.axvline(3.0, color="#C1121F", linewidth=1.6, linestyle="--",
                label="Low-variance flag (3.0)")
    ax2.axvline(np.mean(stddevs), color="#1B3A4B", linewidth=1.4, linestyle=":",
                label=f"Dataset mean ({np.mean(stddevs):.1f})")
    ax2.set_xlabel("Std deviation of turn lengths (words)")
    ax2.set_ylabel("Number of conversations")
    ax2.set_title("Turn-Length Variance per Conversation\n(higher = more natural rhythm)")
    ax2.legend(fontsize=9)

    # ── 2c: histogram customer turn count ─────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    bins = range(int(min(nturns)), int(max(nturns)) + 2)
    ax3.hist(nturns, bins=bins, color=C, edgecolor="white", linewidth=0.7,
             align="left", rwidth=0.8)
    ax3.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax3.set_xlabel("Number of customer turns")
    ax3.set_ylabel("Number of conversations")
    ax3.set_title("Customer Turn Count per Conversation")

    # ── 2d: side-by-side box plots of p10 / median / p90 ──────────────
    ax4 = fig.add_subplot(gs[1, 1])
    data_to_plot = [p10s, p50s, p90s]
    bp = ax4.boxplot(data_to_plot, patch_artist=True, widths=0.5,
                     medianprops=dict(color="white", linewidth=2))
    colors_bp = ["#A8DADC", "#457B9D", "#1B3A4B"]
    for patch, col in zip(bp["boxes"], colors_bp):
        patch.set_facecolor(col)
    ax4.set_xticks([1, 2, 3])
    ax4.set_xticklabels(["P10\n(shortest turns)", "Median", "P90\n(longest turns)"])
    ax4.set_ylabel("Words")
    ax4.set_title("Turn-Length Percentile Distributions\n(box = IQR across conversations)")

    fig.tight_layout()
    _save(fig, "02_structural_stats.png")


# ── Figure 3 — Lexical Signals ────────────────────────────────────────────────
def fig_lexical_signals(results: list):
    """
    Four panels examining hedging vs over-certainty patterns (Wang et al. 2025).
    """
    _set_style()
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("B – Lexical Signals: Hedging vs Over-Certainty\n"
                 "LLM-generated users tend to speak with more certainty than real humans "
                 "(Wang et al. 2025)",
                 fontsize=14, fontweight="bold", y=1.01)
    gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38)

    states = sorted(set(r.emotional_state for r in results))
    state_colors = {s: PALETTE.get(s, "#888888") for s in states}

    # ── 3a: scatter hedge vs certainty, coloured by emotional state ───
    ax1 = fig.add_subplot(gs[0, 0])
    for state in states:
        subset = [r for r in results if r.emotional_state == state]
        ax1.scatter([r.hedge_rate     for r in subset],
                    [r.certainty_rate for r in subset],
                    color=state_colors[state], label=state.replace("_", " "),
                    alpha=0.75, s=55, edgecolors="white", linewidths=0.5)

    # Reference line hedge = certainty
    lim = max(max(r.hedge_rate for r in results),
              max(r.certainty_rate for r in results)) * 1.1
    ax1.plot([0, lim], [0, lim], color="#C1121F", linewidth=1.4, linestyle="--",
             label="hedge = certainty")
    ax1.fill_between([0, lim], [0, 0], [0, lim],
                     alpha=0.06, color="#C1121F", label="Certainty zone (LLM-like)")
    ax1.fill_between([0, lim], [0, lim], [lim, lim],
                     alpha=0.06, color="#2A9D8F", label="Hedging zone (human-like)")
    ax1.set_xlim(0, lim)
    ax1.set_ylim(0, lim)
    ax1.set_xlabel("Hedging rate (phrases per turn)")
    ax1.set_ylabel("Certainty rate (phrases per turn)")
    ax1.set_title("Hedge vs Certainty Rate\n(coloured by emotional state)")
    ax1.legend(fontsize=8, loc="upper right", framealpha=0.85)

    # ── 3b: histogram of hedge − certainty ────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    hmc = [r.hedge_minus_certainty for r in results]
    ax2.hist(hmc, bins=18, color=PALETTE["lexical"], edgecolor="white", linewidth=0.7)
    ax2.axvline(0, color="#C1121F", linewidth=1.8, linestyle="--",
                label="Hedge = Certainty")
    ax2.axvline(np.mean(hmc), color="#1B3A4B", linewidth=1.4, linestyle=":",
                label=f"Dataset mean ({np.mean(hmc):.2f})")
    neg = sum(1 for x in hmc if x < 0)
    ax2.text(0.97, 0.97, f"{neg}/{len(hmc)} convs\ncertainty > hedge",
             transform=ax2.transAxes, ha="right", va="top", fontsize=9,
             color="#C1121F", bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                        edgecolor="#C1121F", alpha=0.85))
    ax2.set_xlabel("Hedging rate − Certainty rate")
    ax2.set_ylabel("Number of conversations")
    ax2.set_title("Hedge − Certainty Distribution\n(positive = more human-like)")
    ax2.legend(fontsize=9)

    # ── 3c: grouped bar — avg hedge & certainty per emotional state ───
    ax3 = fig.add_subplot(gs[1, 0])
    x    = np.arange(len(states))
    w    = 0.35
    avg_h = [np.mean([r.hedge_rate     for r in results if r.emotional_state == s])
             for s in states]
    avg_c = [np.mean([r.certainty_rate for r in results if r.emotional_state == s])
             for s in states]
    bars_h = ax3.bar(x - w/2, avg_h, w, label="Hedge rate",
                     color=[state_colors[s] for s in states], edgecolor="white")
    bars_c = ax3.bar(x + w/2, avg_c, w, label="Certainty rate",
                     color=[state_colors[s] for s in states],
                     edgecolor="white", alpha=0.5, hatch="//")
    ax3.set_xticks(x)
    ax3.set_xticklabels([s.replace("_", "\n") for s in states], fontsize=9)
    ax3.set_ylabel("Phrases per turn")
    ax3.set_title("Avg Hedge & Certainty Rate\nby Emotional State")
    ax3.legend(fontsize=9)

    # ── 3d: stacked bar — certainty > hedge proportion per state ──────
    ax4 = fig.add_subplot(gs[1, 1])
    prop_ok  = []
    prop_bad = []
    for s in states:
        subset = [r for r in results if r.emotional_state == s]
        bad = sum(1 for r in subset if r.certainty_rate > r.hedge_rate)
        prop_bad.append(bad)
        prop_ok.append(len(subset) - bad)

    ax4.bar(states, prop_ok,  label="Hedge ≥ Certainty (human-like)",
            color=PALETTE["density"],   edgecolor="white")
    ax4.bar(states, prop_bad, bottom=prop_ok, label="Certainty > Hedge (LLM-like)",
            color=PALETTE["breadth"],   edgecolor="white")
    ax4.set_xticklabels([s.replace("_", "\n") for s in states], fontsize=9)
    ax4.set_ylabel("Number of conversations")
    ax4.set_title("Hedge vs Certainty Outcome\nby Emotional State")
    ax4.legend(fontsize=9)

    fig.tight_layout()
    _save(fig, "03_lexical_signals.png")


# ── Figure 4 — First-turn Info Density ───────────────────────────────────────
def fig_info_density(results: list):
    """
    Two panels: overall facts-in-turn-1 distribution, and breakdown by goal_clarity.
    """
    _set_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("C – First-Turn Information Density\n"
                 "How much of {merchant, amount, date} the customer volunteers in their "
                 "very first message",
                 fontsize=14, fontweight="bold", y=1.04)

    # ── 4a: overall distribution ──────────────────────────────────────
    ax1 = axes[0]
    facts_dist = defaultdict(int)
    for r in results:
        facts_dist[r.facts_in_turn1] += 1

    bar_labels = ["0 facts\n(human-like)", "1 fact", "2 facts", "3 facts\n(LLM-like)"]
    bar_colors = [PALETTE["density"], PALETTE["lexical"],
                  PALETTE["breadth"], PALETTE["escalating"]]
    counts_4 = [facts_dist.get(i, 0) for i in range(4)]
    bars = ax1.bar(range(4), counts_4, color=bar_colors,
                   edgecolor="white", linewidth=0.7, width=0.6)
    for bar, val in zip(bars, counts_4):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     str(val), ha="center", va="bottom", fontweight="bold")
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(bar_labels)
    ax1.set_ylabel("Number of conversations")
    ax1.set_title("Facts Volunteered in Customer Turn 1\n(ground-truth match against scenario metadata)")

    # ── 4b: stacked bar by goal_clarity ──────────────────────────────
    ax2 = axes[1]
    clarity_vals = sorted(set(r.goal_clarity for r in results))
    x = np.arange(len(clarity_vals))
    bottoms = np.zeros(len(clarity_vals))
    fact_colors = [PALETTE["density"], PALETTE["lexical"],
                   PALETTE["breadth"], PALETTE["escalating"]]
    fact_labels = ["0 facts", "1 fact", "2 facts", "3 facts (info dump)"]

    for fact_n, (col, lbl) in enumerate(zip(fact_colors, fact_labels)):
        heights = []
        for cl in clarity_vals:
            subset = [r for r in results if r.goal_clarity == cl]
            heights.append(sum(1 for r in subset if r.facts_in_turn1 == fact_n))
        ax2.bar(x, heights, bottom=bottoms, color=col, edgecolor="white",
                linewidth=0.6, label=lbl)
        bottoms += np.array(heights, dtype=float)

    ax2.set_xticks(x)
    ax2.set_xticklabels([c.replace("_", "\n") for c in clarity_vals])
    ax2.set_ylabel("Number of conversations")
    ax2.set_title("Info Density in Turn 1\nby Goal Clarity Persona")
    ax2.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    _save(fig, "04_info_density.png")


# ── Figure 5 — Breadth-First Bundling ────────────────────────────────────────
def fig_breadth_first(results: list):
    """
    Three panels: max categories per turn, multi-category turn count, and
    a donut for flagged vs clean.
    """
    _set_style()
    fig = plt.figure(figsize=(14, 5))
    fig.suptitle("D – Breadth-First Topic Bundling\n"
                 "LLM-generated users raise multiple concerns in one turn; "
                 "real users tend to tackle one topic at a time",
                 fontsize=14, fontweight="bold", y=1.04)
    gs = GridSpec(1, 3, figure=fig, wspace=0.38)
    C = PALETTE["breadth"]

    max_cats  = [r.max_categories_per_turn   for r in results]
    multi_ts  = [r.turns_with_multi_category for r in results]
    n_flagged = sum(1 for r in results if r.breadth_first_flag)
    n_clean   = len(results) - n_flagged

    # ── 5a: histogram max categories per turn ─────────────────────────
    ax1 = fig.add_subplot(gs[0])
    bins = range(0, max(max_cats) + 2)
    ax1.hist(max_cats, bins=bins, color=C, edgecolor="white",
             linewidth=0.7, rwidth=0.8, align="left")
    ax1.axvline(2.5, color="#C1121F", linewidth=1.6, linestyle="--",
                label="Flag threshold (≥3)")
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax1.set_xlabel("Concern categories in single turn")
    ax1.set_ylabel("Number of conversations")
    ax1.set_title("Max Concern Categories\nin One Customer Turn")
    ax1.legend(fontsize=9)

    # ── 5b: histogram multi-category turn count ────────────────────────
    ax2 = fig.add_subplot(gs[1])
    bins2 = range(0, max(multi_ts) + 2)
    ax2.hist(multi_ts, bins=bins2, color=C, edgecolor="white",
             linewidth=0.7, rwidth=0.8, align="left")
    ax2.axvline(1.5, color="#C1121F", linewidth=1.6, linestyle="--",
                label="Flag threshold (≥2 turns)")
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax2.set_xlabel("Turns with 2+ concern categories")
    ax2.set_ylabel("Number of conversations")
    ax2.set_title("Multi-Category Turns\nper Conversation")
    ax2.legend(fontsize=9)

    # ── 5c: donut — flagged vs clean ─────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    wedges, texts, autotexts = ax3.pie(
        [n_clean, n_flagged],
        labels=["Clean", "Breadth-first\nflagged"],
        colors=[PALETTE["density"], PALETTE["breadth"]],
        autopct="%1.0f%%",
        startangle=90,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
        textprops=dict(fontsize=10),
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax3.set_title(f"Flagged vs Clean\n({n_flagged}/{len(results)} breadth-first)")

    fig.tight_layout()
    _save(fig, "05_breadth_first_bundling.png")


# ── Figure 6 — Persona Consistency ───────────────────────────────────────────
def fig_persona_consistency(results: list):
    """
    Three panels examining persona-consistency signals:
    turn length by communication style, hedge rate by emotional state,
    and a bar of each consistency warning type.
    """
    _set_style()
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("E – Persona Consistency Signals\n"
                 "Do generated personas actually behave according to their assigned attributes?",
                 fontsize=14, fontweight="bold", y=1.01)
    gs = GridSpec(2, 3, figure=fig, hspace=0.44, wspace=0.38)

    # ── 6a: box plot — avg turn length by communication style ─────────
    ax1 = fig.add_subplot(gs[0, 0])
    styles = sorted(set(r.communication_style for r in results))
    data_by_style = [[r.words_mean for r in results if r.communication_style == s]
                     for s in styles]
    bp = ax1.boxplot(data_by_style, patch_artist=True,
                     medianprops=dict(color="white", linewidth=2.2), widths=0.45)
    for patch, s in zip(bp["boxes"], styles):
        patch.set_facecolor(PALETTE.get(s, "#888888"))
    ax1.set_xticks(range(1, len(styles)+1))
    ax1.set_xticklabels([s.capitalize() for s in styles])
    ax1.set_ylabel("Avg words per turn")
    ax1.set_title("Turn Length by Communication Style\n(direct should be shorter)")
    for i, (s, d) in enumerate(zip(styles, data_by_style), 1):
        ax1.text(i, max(d) * 1.03, f"μ={np.mean(d):.1f}",
                 ha="center", fontsize=9, color=PALETTE.get(s, "#333"))

    # ── 6b: box plot — hedge rate by emotional state ──────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    states = sorted(set(r.emotional_state for r in results))
    data_by_state = [[r.hedge_rate for r in results if r.emotional_state == s]
                     for s in states]
    bp2 = ax2.boxplot(data_by_state, patch_artist=True,
                      medianprops=dict(color="white", linewidth=2.2), widths=0.45)
    for patch, s in zip(bp2["boxes"], states):
        patch.set_facecolor(PALETTE.get(s, "#888888"))
    ax2.set_xticks(range(1, len(states)+1))
    ax2.set_xticklabels([s.replace("_", "\n") for s in states], fontsize=9)
    ax2.set_ylabel("Hedging phrases per turn")
    ax2.set_title("Hedge Rate by Emotional State\n(uncertain states should hedge more)")

    # ── 6c: scatter — words_mean vs words_stddev by style ─────────────
    ax3 = fig.add_subplot(gs[0, 2])
    for s in styles:
        subset = [r for r in results if r.communication_style == s]
        ax3.scatter([r.words_mean   for r in subset],
                    [r.words_stddev for r in subset],
                    color=PALETTE.get(s, "#888888"), label=s.capitalize(),
                    alpha=0.75, s=55, edgecolors="white", linewidths=0.5)
    ax3.set_xlabel("Avg words per turn")
    ax3.set_ylabel("Std deviation of turn lengths")
    ax3.set_title("Turn Length vs Variance\nby Communication Style")
    ax3.legend(fontsize=9)

    # ── 6d: box — avg words/turn by knowledge level ───────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    levels = sorted(set(r.knowledge_level for r in results),
                    key=lambda l: ["novice","intermediate","expert"].index(l)
                    if l in ["novice","intermediate","expert"] else 99)
    data_by_level = [[r.words_mean for r in results if r.knowledge_level == lv]
                     for lv in levels]
    bp3 = ax4.boxplot(data_by_level, patch_artist=True,
                      medianprops=dict(color="white", linewidth=2.2), widths=0.45)
    for patch, lv in zip(bp3["boxes"], levels):
        patch.set_facecolor(PALETTE.get(lv, "#888888"))
    ax4.set_xticks(range(1, len(levels)+1))
    ax4.set_xticklabels([lv.capitalize() for lv in levels])
    ax4.set_ylabel("Avg words per turn")
    ax4.set_title("Turn Length by Knowledge Level\n(expert may write longer turns)")

    # ── 6e: bar — persona consistency warning types ───────────────────
    ax5 = fig.add_subplot(gs[1, 1:])
    warn_labels = [
        "Missing frustration\nmarkers",
        "Unexpected frustration\n(calm persona)",
        "Missing strong\nfrustration",
        "Novice used jargon\nbefore agent",
        "Style/length\nmismatch",
    ]
    warn_keys = [
        "MISSING_FRUSTRATION_MARKERS",
        "UNEXPECTED_FRUSTRATION",
        "MISSING_STRONG_FRUSTRATION",
        "NOVICE_JARGON",
        "DIRECT_STYLE_BUT_LONG_TURNS",
    ]
    warn_counts = []
    for wk in warn_keys:
        warn_counts.append(sum(1 for r in results if any(w.startswith(wk) for w in r.warnings)))

    bar_colors = [PALETTE["persona"]] * len(warn_labels)
    bar_colors[1] = "#E9C46A"   # highlight unexpected frustration
    bars = ax5.bar(range(len(warn_labels)), warn_counts, color=bar_colors,
                   edgecolor="white", linewidth=0.7, width=0.55)
    for bar, val in zip(bars, warn_counts):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                 str(val), ha="center", va="bottom", fontweight="bold")
    ax5.set_xticks(range(len(warn_labels)))
    ax5.set_xticklabels(warn_labels, fontsize=9)
    ax5.set_ylabel("Number of conversations flagged")
    ax5.set_title("Persona Consistency Warning Breakdown\nacross all 80 conversations")
    ax5.set_ylim(0, max(warn_counts) * 1.25)

    fig.tight_layout()
    _save(fig, "06_persona_consistency.png")


# ── Figure 7 — Outlier Summary ────────────────────────────────────────────────
def fig_outlier_summary(results: list):
    """
    Two panels: warning-count distribution with threshold annotated, and a
    per-outlier horizontal bar showing which warnings they carry.
    """
    _set_style()
    outliers = sorted([r for r in results if r.is_outlier], key=lambda r: -r.warning_count)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(outliers) * 0.9 + 3.0)))
    fig.suptitle("Outlier Summary — Conversations Flagged on ≥3 Dimensions\n"
                 "These are the highest-priority candidates for manual review or regeneration",
                 fontsize=14, fontweight="bold", y=1.04)

    # ── 7a: histogram of warning counts ──────────────────────────────
    ax1 = axes[0]
    warn_counts = [r.warning_count for r in results]
    bins = range(0, max(warn_counts) + 2)
    ax1.hist(warn_counts, bins=bins, color=PALETTE["structural"],
             edgecolor="white", linewidth=0.7, rwidth=0.8, align="left")
    ax1.axvline(OUTLIER_THRESHOLD - 0.5, color="#C1121F", linewidth=1.8,
                linestyle="--", label=f"Outlier threshold (≥{OUTLIER_THRESHOLD})")
    outlier_n = sum(1 for x in warn_counts if x >= OUTLIER_THRESHOLD)
    ax1.text(OUTLIER_THRESHOLD + 0.05, ax1.get_ylim()[1] * 0.85,
             f"{outlier_n} outlier(s)", color="#C1121F", fontsize=10, fontweight="bold")
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax1.set_xlabel("Number of warnings per conversation")
    ax1.set_ylabel("Number of conversations")
    ax1.set_title("Warning Count Distribution\nacross all 80 conversations")
    ax1.legend(fontsize=9)

    # ── 7b: per-outlier warning breakdown ─────────────────────────────
    ax2 = axes[1]
    if not outliers:
        ax2.text(0.5, 0.5, "No outliers found ✓",
                 ha="center", va="center", fontsize=14,
                 transform=ax2.transAxes, color=PALETTE["density"])
        ax2.set_axis_off()
    else:
        WARN_CATEGORY = {
            "HIGH_AVG_TURN_LENGTH":            ("Structural",  PALETTE["structural"]),
            "LOW_LENGTH_VARIANCE":             ("Structural",  PALETTE["structural"]),
            "CERTAINTY_EXCEEDS_HEDGING":       ("Lexical",     PALETTE["lexical"]),
            "ZERO_HEDGING_FOR_UNCERTAIN":      ("Lexical",     PALETTE["lexical"]),
            "INFO_DUMP_TURN1":                 ("Info density",PALETTE["density"]),
            "BREADTH_FIRST_BUNDLING":          ("Breadth",     PALETTE["breadth"]),
            "MISSING_FRUSTRATION_MARKERS":     ("Persona",     PALETTE["persona"]),
            "UNEXPECTED_FRUSTRATION":          ("Persona",     PALETTE["persona"]),
            "MISSING_STRONG_FRUSTRATION":      ("Persona",     PALETTE["persona"]),
            "NOVICE_JARGON":                   ("Persona",     PALETTE["persona"]),
            "DIRECT_STYLE_BUT_LONG_TURNS":     ("Persona",     PALETTE["persona"]),
            "INDIRECT_STYLE_BUT_VERY_SHORT":   ("Persona",     PALETTE["persona"]),
        }

        y_pos   = range(len(outliers))
        y_labels = [f"Conv {r.conv_idx:03d}\n{r.emotional_state.replace('_',' ')} · "
                    f"{r.knowledge_level} · {r.communication_style}"
                    for r in outliers]

        for i, r in enumerate(outliers):
            x_start = 0
            for w in r.warnings:
                matched_key = next((k for k in WARN_CATEGORY if w.startswith(k)), None)
                label, col = WARN_CATEGORY.get(matched_key, ("Other", "#999999"))
                ax2.barh(i, 1, left=x_start, color=col, edgecolor="white",
                         linewidth=0.8, height=0.6)
                ax2.text(x_start + 0.5, i, label, ha="center", va="center",
                         fontsize=7.5, color="black", fontweight="bold",
                         clip_on=True)
                x_start += 1

        ax2.set_yticks(list(y_pos))
        ax2.set_yticklabels(y_labels, fontsize=9)
        ax2.invert_yaxis()
        ax2.set_xlabel("Warnings (each block = 1 warning)")
        ax2.set_xlim(0, max(r.warning_count for r in outliers) + 0.5)
        ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax2.set_title(f"Warning Composition per Outlier Conversation\n"
                      f"({len(outliers)} outlier(s) with ≥{OUTLIER_THRESHOLD} warnings)")

        legend_items = [mpatches.Patch(color=c, label=lbl) for lbl, c in
                        {v[0]: v[1] for v in WARN_CATEGORY.values()}.items()]
        ax2.legend(handles=legend_items, fontsize=8, loc="lower right",
                   framealpha=0.9, edgecolor="#cccccc")

    fig.tight_layout()
    _save(fig, "07_outlier_summary.png")


def generate_all_visualisations(results: list):
    IMAGES_DIR.mkdir(exist_ok=True)
    print(f"\nGenerating visualisations → {IMAGES_DIR}/")
    fig_warning_summary(results)
    fig_structural_stats(results)
    fig_lexical_signals(results)
    fig_info_density(results)
    fig_breadth_first(results)
    fig_persona_consistency(results)
    fig_outlier_summary(results)
    print(f"  ✓  All 7 figures saved.\n")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Layer 1 rule-based evaluation of synthetic banking conversations.",
    )
    parser.add_argument(
        "--dumps-dir",
        default=None,
        help=(
            "Directory containing conversation_NNN.txt dump files. "
            "Defaults to conversation_dumps/ next to this script."
        ),
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV path. Defaults to eval_results.csv next to this script.",
    )
    args = parser.parse_args()

    dumps_dir  = Path(args.dumps_dir)  if args.dumps_dir  else DUMPS_DIR
    output_csv = Path(args.output_csv) if args.output_csv else OUTPUT_CSV

    dump_files = sorted(dumps_dir.glob("conversation_*.txt"))
    if not dump_files:
        print(f"No conversation dumps found in {dumps_dir}")
        return

    print(f"Parsing {len(dump_files)} conversation dump(s) from {dumps_dir} …")
    conversations = []
    for f in dump_files:
        conv = parse_conversation(f)
        if conv:
            conversations.append(conv)
        else:
            print(f"  ⚠  Could not parse {f.name}")

    print(f"  → {len(conversations)} conversations parsed successfully.")

    print("Running evaluation …")
    results = [evaluate(c) for c in conversations]

    pop_notes = population_checks(results)

    print_summary(results, pop_notes)
    write_csv(results, output_csv)
    generate_all_visualisations(results)

if __name__ == "__main__":
    main()
