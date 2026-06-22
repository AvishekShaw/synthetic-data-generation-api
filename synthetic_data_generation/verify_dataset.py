#!/usr/bin/env python3
"""
verify_dataset.py

Validates the generated JSONL files against all pipeline acceptance criteria.
Exits non-zero if any check fails — suitable as a CI gate before a full run.

Usage:
    python verify_dataset.py \
        --success userlm_success.jsonl \
        --failure userlm_failure.jsonl \
        --type-a  userlm_type_a.jsonl \
        --type-b  userlm_type_b.jsonl \
        [--heldout test_heldout.jsonl]
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


# ─────────────────────────────────────────────
# LOADING
# ─────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARN: {path}:{lineno} — JSON parse error: {e}")
    return rows


# ─────────────────────────────────────────────
# RESULT HELPERS
# ─────────────────────────────────────────────

RESULTS: list[tuple[str, bool, str]] = []  # (check_name, passed, detail)


def check(name: str, passed: bool, detail: str = "") -> bool:
    RESULTS.append((name, passed, detail))
    return passed


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = (len(sorted_vals) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


# ─────────────────────────────────────────────
# SCHEMA INTEGRITY  (Acceptance 6)
# ─────────────────────────────────────────────

def check_schema(rows: list[dict], label: str, require_probing_type: bool = False) -> None:
    bad_rows = 0
    missing_eos = 0
    missing_probing = 0

    for r in rows:
        if not all(k in r for k in ("input", "output", "_meta")):
            bad_rows += 1
            continue
        if not str(r["output"]).endswith("<eos>"):
            missing_eos += 1
        if require_probing_type and "probing_type" not in r.get("_meta", {}):
            missing_probing += 1

    check(f"[schema] {label}: input/output/_meta present",
          bad_rows == 0, f"{bad_rows} rows missing required fields")
    check(f"[schema] {label}: output ends with <eos>",
          missing_eos == 0, f"{missing_eos} rows missing <eos>")
    if require_probing_type:
        check(f"[schema] {label}: probing_type in _meta",
              missing_probing == 0, f"{missing_probing} rows missing probing_type")


def check_no_idx_collisions(all_rows_by_label: dict[str, list[dict]]) -> None:
    seen: dict[int, str] = {}
    collisions = []
    for label, rows in all_rows_by_label.items():
        for r in rows:
            idx = r.get("_meta", {}).get("generation_idx")
            if idx is None:
                continue
            if idx in seen:
                collisions.append(f"idx={idx} in both {seen[idx]} and {label}")
            else:
                seen[idx] = label
    check("[schema] No generation_idx collisions across files",
          len(collisions) == 0,
          f"{len(collisions)} collision(s): {collisions[:3]}" if collisions else "")


# ─────────────────────────────────────────────
# ACCEPTANCE 1 — Content diversity
# ─────────────────────────────────────────────

def parse_amount_dollars(amount_str: str) -> float:
    try:
        return float(re.sub(r"[,$]", "", amount_str))
    except (ValueError, TypeError):
        return 0.0


def parse_date_ordinal(date_str: str) -> int:
    """Return a rough day-of-year ordinal for span calculation."""
    import datetime
    months = {m: i for i, m in enumerate([
        "January","February","March","April","May","June",
        "July","August","September","October","November","December"
    ], 1)}
    parts = str(date_str).split()
    if len(parts) == 2:
        month_num = months.get(parts[0], 0)
        try:
            day = int(parts[1])
            return datetime.date(2024, month_num, min(day, 28)).timetuple().tm_yday
        except (ValueError, TypeError):
            pass
    return 0


def check_content_diversity(rows: list[dict], label: str) -> None:
    merchants: set[str] = set()
    amounts: set[str] = set()
    date_ordinals: list[int] = []

    for r in rows:
        sc = r.get("_meta", {}).get("scenario", {})
        if sc.get("merchant_name"):
            merchants.add(sc["merchant_name"])
        if sc.get("amount"):
            amounts.add(sc["amount"])
        if sc.get("transaction_date"):
            d = parse_date_ordinal(sc["transaction_date"])
            if d:
                date_ordinals.append(d)

    date_span = max(date_ordinals) - min(date_ordinals) if len(date_ordinals) >= 2 else 0

    check(f"[diversity] {label}: ≥40 distinct merchants",
          len(merchants) >= 40, f"got {len(merchants)}")
    check(f"[diversity] {label}: ≥100 distinct amounts",
          len(amounts) >= 100, f"got {len(amounts)}")
    check(f"[diversity] {label}: date span ≥120 days",
          date_span >= 120, f"got ~{date_span} days")


# ─────────────────────────────────────────────
# ACCEPTANCE 2 — Decorrelation
# ─────────────────────────────────────────────

def check_failure_decorrelation(rows: list[dict]) -> None:
    all_modes: set[str] = set()
    persona_to_modes: dict[str, set[str]] = defaultdict(set)
    seen_convos: set[int] = set()

    for r in rows:
        meta = r.get("_meta", {})
        gidx = meta.get("generation_idx")
        if gidx in seen_convos:
            continue
        seen_convos.add(gidx)

        fm = meta.get("failure_mode")
        if not fm:
            continue
        all_modes.add(fm)
        p = meta.get("persona", {})
        pk = f"{p.get('knowledge_level')}/{p.get('communication_style')}"
        persona_to_modes[pk].add(fm)

    check("[corr] failure: all 8 modes present",
          len(all_modes) >= 8, f"got {sorted(all_modes)}")

    min_modes = min((len(v) for v in persona_to_modes.values()), default=0)
    check("[corr] failure: each persona co-occurs with ≥3 failure modes",
          min_modes >= 3, f"min across personas = {min_modes}")


def check_type_a_decorrelation(rows: list[dict]) -> None:
    scenario_to_beliefs: dict[str, set[str]] = defaultdict(set)
    seen_convos: set[int] = set()

    for r in rows:
        meta = r.get("_meta", {})
        gidx = meta.get("generation_idx")
        if gidx in seen_convos:
            continue
        seen_convos.add(gidx)

        belief = meta.get("wrong_prior_belief", "")
        sc = meta.get("scenario", {})
        sk = f"{sc.get('certainty')}/{sc.get('information_completeness')}"
        if belief:
            scenario_to_beliefs[sk].add(belief)

    min_beliefs = min((len(v) for v in scenario_to_beliefs.values()), default=0)
    check("[corr] type_a: each structural scenario co-occurs with ≥3 prior beliefs",
          min_beliefs >= 3, f"min across scenarios = {min_beliefs}")


def check_type_b_decorrelation(rows: list[dict]) -> None:
    scenario_to_errors: dict[str, set[str]] = defaultdict(set)
    seen_convos: set[int] = set()

    for r in rows:
        meta = r.get("_meta", {})
        gidx = meta.get("generation_idx")
        if gidx in seen_convos:
            continue
        seen_convos.add(gidx)

        err = meta.get("agent_failure_mode", "")
        sc = meta.get("scenario", {})
        sk = f"{sc.get('certainty')}/{sc.get('information_completeness')}"
        if err:
            scenario_to_errors[sk].add(err)

    min_errors = min((len(v) for v in scenario_to_errors.values()), default=0)
    check("[corr] type_b: each structural scenario co-occurs with ≥3 planted-error modes",
          min_errors >= 3, f"min across scenarios = {min_errors}")


# ─────────────────────────────────────────────
# ACCEPTANCE 3 — Length distribution
# ─────────────────────────────────────────────

def user_msg_words(rows: list[dict]) -> list[int]:
    """Extract word counts from non-terminal user turns (output != '<eos>')."""
    counts = []
    for r in rows:
        out = r.get("output", "")
        if out != "<eos>" and out.endswith("<eos>"):
            msg = out[:-5].strip()
            counts.append(len(msg.split()))
    return counts


def check_length_distribution(rows: list[dict], label: str) -> None:
    counts = sorted(user_msg_words(rows))
    if not counts:
        check(f"[length] {label}: has user messages", False, "no messages found")
        return

    n = len(counts)
    median = percentile(counts, 50)
    p90 = percentile(counts, 90)
    pct_over_30 = sum(1 for c in counts if c > 30) / n * 100
    pct_one_word = sum(1 for c in counts if c <= 1) / n * 100

    check(f"[length] {label}: median ≥12 words", median >= 12,
          f"median={median:.1f}")
    check(f"[length] {label}: p90 ≥35 words", p90 >= 35,
          f"p90={p90:.1f}")
    check(f"[length] {label}: ≥10% msgs >30 words", pct_over_30 >= 10,
          f"{pct_over_30:.1f}%")
    check(f"[length] {label}: <5% one-word msgs", pct_one_word < 5,
          f"{pct_one_word:.1f}%")


# ─────────────────────────────────────────────
# ACCEPTANCE 4 — Balance
# ─────────────────────────────────────────────

def check_balance(counts: dict[str, int], target: int | None = None) -> None:
    if not counts:
        return
    vals = list(counts.values())
    mean = sum(vals) / len(vals)
    ref = target or mean
    worst_pct = max(abs(v - ref) / ref * 100 for v in vals)
    check("[balance] per-type conversation counts within ±15% of mean",
          worst_pct <= 15,
          f"worst deviation = {worst_pct:.1f}% | counts = {counts}")


# ─────────────────────────────────────────────
# ACCEPTANCE 5 — Held-out disjointness
# ─────────────────────────────────────────────

def check_heldout_disjoint(train_rows: list[dict], heldout_rows: list[dict]) -> None:
    train_merchants: set[str] = set()
    for r in train_rows:
        m = r.get("_meta", {}).get("scenario", {}).get("merchant_name", "")
        if m:
            train_merchants.add(m)

    heldout_merchants: set[str] = set()
    for r in heldout_rows:
        m = r.get("_meta", {}).get("scenario", {}).get("merchant_name", "")
        if m:
            heldout_merchants.add(m)

    overlap = train_merchants & heldout_merchants
    check("[heldout] 0 merchants shared between training and held-out",
          len(overlap) == 0,
          f"overlap={sorted(overlap)}" if overlap else "")


# ─────────────────────────────────────────────
# CONVERSATION COUNT HELPERS
# ─────────────────────────────────────────────

def count_conversations(rows: list[dict]) -> int:
    """Count distinct generation_idx values (= distinct conversations)."""
    return len({r.get("_meta", {}).get("generation_idx") for r in rows} - {None})


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify generated JSONL datasets against pipeline acceptance criteria.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--success", default=None, help="userlm_success.jsonl path")
    parser.add_argument("--failure", default=None, help="userlm_failure.jsonl path")
    parser.add_argument("--type-a",  default=None, dest="type_a", help="userlm_type_a.jsonl path")
    parser.add_argument("--type-b",  default=None, dest="type_b", help="userlm_type_b.jsonl path")
    parser.add_argument("--heldout", default=None, help="test_heldout.jsonl path")
    parser.add_argument("--target-conversations", type=int, default=None, dest="target_conversations",
                        help="Expected per-type conversation count for balance check.")
    args = parser.parse_args()

    files = {
        "success": args.success,
        "failure": args.failure,
        "type_a":  args.type_a,
        "type_b":  args.type_b,
    }
    provided = {k: v for k, v in files.items() if v and Path(v).exists()}

    if not provided:
        print("No input files found. Pass at least one of --success/--failure/--type-a/--type-b.")
        sys.exit(1)

    # Load
    data: dict[str, list[dict]] = {}
    for label, path in provided.items():
        print(f"Loading {label}: {path} ...")
        data[label] = load_jsonl(path)
        print(f"  {len(data[label])} rows, {count_conversations(data[label])} conversations")

    heldout_rows: list[dict] = []
    if args.heldout and Path(args.heldout).exists():
        print(f"Loading heldout: {args.heldout} ...")
        heldout_rows = load_jsonl(args.heldout)
        print(f"  {len(heldout_rows)} rows, {count_conversations(heldout_rows)} conversations")

    print()

    # ── Schema integrity ─────────────────────────────────────────────────────
    for label, rows in data.items():
        check_schema(rows, label, require_probing_type=(label in ("type_a", "type_b")))
    if heldout_rows:
        check_schema(heldout_rows, "heldout")

    all_for_collision = dict(data)
    if heldout_rows:
        all_for_collision["heldout"] = heldout_rows
    check_no_idx_collisions(all_for_collision)

    # ── Acceptance 1: content diversity ──────────────────────────────────────
    for label, rows in data.items():
        check_content_diversity(rows, label)

    # ── Acceptance 2: decorrelation ──────────────────────────────────────────
    if "failure" in data:
        check_failure_decorrelation(data["failure"])
    if "type_a" in data:
        check_type_a_decorrelation(data["type_a"])
    if "type_b" in data:
        check_type_b_decorrelation(data["type_b"])

    # ── Acceptance 3: length distribution ────────────────────────────────────
    all_train = [r for rows in data.values() for r in rows]
    check_length_distribution(all_train, "combined training")
    for label, rows in data.items():
        check_length_distribution(rows, label)

    # ── Acceptance 4: balance ────────────────────────────────────────────────
    convo_counts = {label: count_conversations(rows) for label, rows in data.items()}
    check_balance(convo_counts, target=args.target_conversations)

    # ── Acceptance 5: held-out disjointness ──────────────────────────────────
    if heldout_rows:
        check_heldout_disjoint(all_train, heldout_rows)
    else:
        print("  (skipping heldout disjointness check — no --heldout file provided)")

    # ── Print results table ──────────────────────────────────────────────────
    print()
    print("=" * 72)
    print(f"{'CHECK':<52} {'RESULT':<8} DETAIL")
    print("=" * 72)
    failures = 0
    for name, passed, detail in RESULTS:
        status = "PASS" if passed else "FAIL"
        if not passed:
            failures += 1
        suffix = f"  {detail}" if detail else ""
        print(f"  {name:<50} {status}{suffix}")

    print("=" * 72)
    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else f'{failures} CHECK(S) FAILED'}\n")
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
