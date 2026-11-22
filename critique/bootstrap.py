"""95% bootstrap intervals from per-question scores (output of score_per_question.py). CPU only, seconds.

One file  -> each metric's mean and 95% interval.
Two files -> also the paired difference (new minus base) with its 95% interval. Paired = the same resampled
             questions are used for both runs, which is the fair way to compare two systems on one test set.

    python critique/bootstrap.py --base out/asqa_keeptok_0_end.asqa_A.perq.jsonl
    python critique/bootstrap.py --base C1.perq.jsonl --new fix1.perq.jsonl --log logs/asqa_123.log

How to read it: if the interval of the difference contains 0, the change is not shown to be real on this split.
Decoding is greedy, so there is no seed noise; the interval only reflects which questions happened to be picked.
"""
import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_split_indices  # noqa: E402

METRICS = ["str_em", "rougeLsum", "citation_rec", "citation_prec", "length"]


def load(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return {r["q_idx"]: r for r in rows}


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def interval(values_fn, q_ids, n_boot, rng):
    """values_fn(list of q_ids) -> statistic. Returns (point, low, high)."""
    point = values_fn(q_ids)
    stats = sorted(values_fn([rng.choice(q_ids) for _ in q_ids]) for _ in range(n_boot))
    return point, stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot) - 1]


def seconds_per_question(log_path):
    """Reads the 'Done: N items in M min' line that run_long_form_static.py prints."""
    for line in open(log_path):
        m = re.search(r"Done: (\d+) items in ([\d.]+) min", line)
        if m:
            return 60.0 * float(m.group(2)) / int(m.group(1))
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="per-question file of the reference run (e.g. C1)")
    parser.add_argument("--new", default=None, help="per-question file of the run to compare")
    parser.add_argument("--split", default="", help="comma-separated split files: only these questions")
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log", default=None, help="inference log of --new (or --base) for seconds per question")
    parser.add_argument("--out", default=None, help="optional JSON file for the numbers")
    args = parser.parse_args()

    base = load(args.base)
    new = load(args.new) if args.new else None
    q_ids = sorted(base) if new is None else sorted(set(base) & set(new))
    if new is not None and len(q_ids) != len(base):
        print(f"warning: only {len(q_ids)} questions are in both files")
    if args.split:
        wanted = set(load_split_indices(args.split))
        q_ids = [q for q in q_ids if q in wanted]
        assert len(q_ids) == len(wanted), f"{len(wanted) - len(q_ids)} split questions are missing from the files"
    rng = random.Random(args.seed)
    result = {"n_questions": len(q_ids), "n_boot": args.n_boot}

    print(f"{'metric':15s} {'base':>22s}" + (f" {'new':>22s} {'new - base':>24s}" if new else ""))
    for m in METRICS:
        def stat(rows, ids):
            return mean([rows[q][m] for q in ids if rows[q].get(m) is not None])
        if all(base[q].get(m) is None for q in q_ids):
            continue
        b = interval(lambda ids: stat(base, ids), q_ids, args.n_boot, rng)
        line = f"{m:15s} {b[0]:7.2f} [{b[1]:6.2f}, {b[2]:6.2f}]"
        result[m] = {"base": b}
        if new is not None:
            n = interval(lambda ids: stat(new, ids), q_ids, args.n_boot, rng)
            d = interval(lambda ids: stat(new, ids) - stat(base, ids), q_ids, args.n_boot, rng)
            line += f" {n[0]:7.2f} [{n[1]:6.2f}, {n[2]:6.2f}] {d[0]:+7.2f} [{d[1]:+6.2f}, {d[2]:+6.2f}]"
            line += "  *" if d[1] > 0 or d[2] < 0 else ""
            result[m].update({"new": n, "diff": d})
        print(line)
    if new is not None:
        print("* = the 95% interval of the difference excludes 0")
    if args.log:
        result["seconds_per_question"] = seconds_per_question(args.log)
        print("seconds per question:", result["seconds_per_question"])
    if args.out:
        json.dump(result, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
