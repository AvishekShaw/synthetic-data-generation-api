#!/usr/bin/env python3
"""
split_dataset.py

Combines multiple UserLM JSONL datasets and produces a single leakage-free
train / val split. The split is done at the CONVERSATION level (by
generation_idx), not the example level — so every training example from a
given conversation ends up in the same partition.

Supported dataset types (auto-detected from _meta fields):
  success  — goal_completed=True,  no probing_type     (generate_conversations.py --mode success)
  failure  — goal_completed=False, no probing_type     (generate_conversations.py --mode failure)
  type_a   — probing_type=inadvertent                  (generate_probe_conversations.py --mode type_a)
  type_b   — probing_type=adversarial                  (generate_probe_conversations.py --mode type_b)

Usage:
    python split_dataset.py \\
        --files userlm_data.jsonl \\
                failure_conversations.jsonl \\
                probe_conversations_type_a.jsonl \\
                probe_conversations_type_b.jsonl \\
        --ratio 0.8 \\
        --seed 42 \\
        --train-out train_combined.jsonl \\
        --val-out   val_combined.jsonl
"""

import json
import random
import argparse
from collections import defaultdict
from pathlib import Path


# ─────────────────────────────────────────────
# DATASET TYPE DETECTION
# ─────────────────────────────────────────────

def detect_type(meta: dict) -> str:
    """Infer dataset type from _meta fields."""
    pt = meta.get("probing_type")
    if pt == "inadvertent":
        return "type_a"
    if pt == "adversarial":
        return "type_b"
    if meta.get("goal_completed") is False:
        return "failure"
    return "success"


# ─────────────────────────────────────────────
# LOADING
# ─────────────────────────────────────────────

def load_files(paths: list[str]) -> dict[int, list[dict]]:
    """
    Load all JSONL files. Group entries by generation_idx.
    Raises if the same generation_idx appears in two different files
    (collision between datasets — fix idx_offset in the generators).
    """
    by_idx: dict[int, list[dict]] = defaultdict(list)
    seen_in_file: dict[int, str] = {}   # idx → filename where first seen

    for path in paths:
        p = Path(path).expanduser()
        if not p.exists():
            print(f"  ⚠  File not found, skipping: {path}")
            continue

        count = 0
        with open(p) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  ⚠  {p.name}:{lineno} — JSON parse error: {e}")
                    continue

                idx = entry.get("_meta", {}).get("generation_idx")
                if idx is None:
                    print(f"  ⚠  {p.name}:{lineno} — missing generation_idx, skipping entry")
                    continue

                # Collision guard
                if idx in seen_in_file and seen_in_file[idx] != str(p):
                    raise ValueError(
                        f"generation_idx {idx} appears in both "
                        f"'{seen_in_file[idx]}' and '{p}'. "
                        f"Fix idx_offset in the generator scripts."
                    )
                seen_in_file[idx] = str(p)
                by_idx[idx].append(entry)
                count += 1

        print(f"  Loaded {count:>5} entries from {p.name}")

    return dict(by_idx)


# ─────────────────────────────────────────────
# SPLIT
# ─────────────────────────────────────────────

def split_conversations(
    by_idx: dict[int, list[dict]],
    ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """
    Shuffle conversation indices and split into train / val.
    Returns (train_indices, val_indices).
    """
    all_indices = sorted(by_idx.keys())
    rng = random.Random(seed)
    rng.shuffle(all_indices)
    split = max(1, int(len(all_indices) * ratio))
    return all_indices[:split], all_indices[split:]


# ─────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────

def compute_stats(indices: list[int], by_idx: dict[int, list[dict]]) -> dict:
    """Compute breakdown stats over a set of conversation indices."""
    stats: dict = {
        "conversations": len(indices),
        "entries":       0,
        "by_type":       defaultdict(int),
        "by_failure_mode":    defaultdict(int),
        "by_agent_failure":   defaultdict(int),
        "goal_completed_true":  0,
        "goal_completed_false": 0,
        "goal_completed_unknown": 0,
        "user_caught_error_true":  0,
        "user_caught_error_false": 0,
    }

    for idx in indices:
        entries = by_idx[idx]
        # Use the first entry's _meta to classify the conversation
        meta = entries[0].get("_meta", {})
        dtype = detect_type(meta)
        stats["by_type"][dtype] += 1

        fm = meta.get("failure_mode")
        if fm:
            stats["by_failure_mode"][fm] += 1

        af = meta.get("agent_failure_mode")
        if af:
            stats["by_agent_failure"][af] += 1

        gc = meta.get("goal_completed")
        if gc is True:
            stats["goal_completed_true"] += 1
        elif gc is False:
            stats["goal_completed_false"] += 1
        else:
            stats["goal_completed_unknown"] += 1

        uce = meta.get("user_caught_error")
        if uce is True:
            stats["user_caught_error_true"] += 1
        elif uce is False:
            stats["user_caught_error_false"] += 1

        stats["entries"] += len(entries)

    return stats


def print_stats(label: str, stats: dict) -> None:
    w = 42
    print(f"\n  {'─' * w}")
    print(f"  {label}")
    print(f"  {'─' * w}")
    print(f"  Conversations : {stats['conversations']}")
    print(f"  Entries       : {stats['entries']}")

    print(f"\n  By dataset type:")
    for t in ["success", "failure", "type_a", "type_b"]:
        cnt = stats["by_type"].get(t, 0)
        if cnt:
            print(f"    {t:<12} : {cnt} conversations")

    if stats["by_failure_mode"]:
        print(f"\n  Failure modes (user dropout):")
        for fm, cnt in sorted(stats["by_failure_mode"].items()):
            print(f"    {fm:<20} : {cnt}")

    if stats["by_agent_failure"]:
        print(f"\n  Planted agent errors (type_b):")
        for af, cnt in sorted(stats["by_agent_failure"].items()):
            print(f"    {af:<20} : {cnt}")

    if stats["goal_completed_true"] or stats["goal_completed_false"]:
        print(f"\n  Goal completed:")
        print(f"    True    : {stats['goal_completed_true']}")
        print(f"    False   : {stats['goal_completed_false']}")
        if stats["goal_completed_unknown"]:
            print(f"    Unknown : {stats['goal_completed_unknown']}")

    if stats["user_caught_error_true"] or stats["user_caught_error_false"]:
        print(f"\n  User caught planted error:")
        print(f"    True    : {stats['user_caught_error_true']}")
        print(f"    False   : {stats['user_caught_error_false']}")


# ─────────────────────────────────────────────
# WRITE
# ─────────────────────────────────────────────

def write_split(
    indices: list[int],
    by_idx: dict[int, list[dict]],
    out_path: Path,
    shuffle_entries: bool = True,
    seed: int = 42,
) -> None:
    """Collect all entries for the given indices, optionally shuffle, and write."""
    entries: list[dict] = []
    for idx in indices:
        entries.extend(by_idx[idx])

    if shuffle_entries:
        rng = random.Random(seed + 1)
        rng.shuffle(entries)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine and split UserLM datasets into train/val.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--files", nargs="+", required=True, metavar="JSONL",
        help=(
            "One or more JSONL files to combine. Typical order: "
            "success, failure, type_a, type_b."
        ),
    )
    parser.add_argument(
        "--ratio", type=float, default=0.8,
        help="Fraction of conversations to put in train (default: 0.8 = 80/20 split).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible shuffling (default: 42).",
    )
    parser.add_argument(
        "--train-out", default="train_combined.jsonl",
        help="Output path for training set (default: train_combined.jsonl).",
    )
    parser.add_argument(
        "--val-out", default="val_combined.jsonl",
        help="Output path for validation set (default: val_combined.jsonl).",
    )
    args = parser.parse_args()

    if not (0.0 < args.ratio < 1.0):
        parser.error("--ratio must be between 0 and 1 (exclusive).")

    print(f"\n=== UserLM Dataset Splitter ===\n")
    print(f"Files   : {', '.join(args.files)}")
    print(f"Ratio   : {args.ratio:.0%} train / {1 - args.ratio:.0%} val")
    print(f"Seed    : {args.seed}")
    print(f"Train → : {args.train_out}")
    print(f"Val   → : {args.val_out}")
    print()

    # ── Load ──────────────────────────────────────────────────
    print("Loading files:")
    by_idx = load_files(args.files)

    if not by_idx:
        print("\nNo entries loaded. Check your file paths.")
        return

    total_convs = len(by_idx)
    total_entries = sum(len(v) for v in by_idx.values())
    print(f"\n  Total: {total_entries} entries across {total_convs} conversations\n")

    # ── Split ─────────────────────────────────────────────────
    train_idx, val_idx = split_conversations(by_idx, args.ratio, args.seed)

    # ── Stats ─────────────────────────────────────────────────
    overall_stats = compute_stats(list(by_idx.keys()), by_idx)
    train_stats   = compute_stats(train_idx, by_idx)
    val_stats     = compute_stats(val_idx,   by_idx)

    print_stats("OVERALL", overall_stats)
    print_stats(f"TRAIN  ({args.ratio:.0%})", train_stats)
    print_stats(f"VAL    ({1 - args.ratio:.0%})", val_stats)

    # ── Write ─────────────────────────────────────────────────
    train_path = Path(args.train_out).expanduser()
    val_path   = Path(args.val_out).expanduser()

    write_split(train_idx, by_idx, train_path, shuffle_entries=True,  seed=args.seed)
    write_split(val_idx,   by_idx, val_path,   shuffle_entries=False, seed=args.seed)

    print(f"\n{'=' * 44}")
    print(f"  Written {train_stats['entries']:>5} entries → {train_path.name}")
    print(f"  Written {val_stats['entries']:>5} entries → {val_path.name}")
    print(f"{'=' * 44}\n")


if __name__ == "__main__":
    main()
