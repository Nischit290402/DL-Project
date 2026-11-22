"""H1 of contradiction/PREREG_C.md: is disagreement among a question's candidates a sign that the chosen answer is
wrong? (laptop, Python standard library only)

Disagreement D_q = mean over the 10 pairs of depth-1 candidates of max(p_con(i->j), p_con(j->i)), variant noctx.
Questions are split at the median of D_q (lower half = the floor(n/2) least-disagreeing questions, ties broken by
question index). Primary: mean str-em of the chosen answer, low half minus high half, with a 95% bootstrap interval
(questions resampled, median split redone in each resample). Secondary: the same for citation recall and ROUGE-Lsum,
and the AUROC of D_q for "chosen str-em = 0".

    python3 contradiction/uncertainty_test.py --cands results/runs/asqa_cands_C.json \
        --nli results/runs/cand_nli_C.jsonl --perq results/runs/rr_released_C.json.perq.jsonl \
        --split critique/splits/asqa_C.json --out results/uncertainty_C.json
"""
import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "critique"))
from common import load_split_indices  # noqa: E402

SEED = 6103
METRICS = ("str_em", "citation_rec", "rougeLsum")


def disagreement(cands_path, nli_path, keep):
    p = collections.defaultdict(dict)
    for line in open(nli_path):
        r = json.loads(line)
        if r["variant"] == "noctx" and r["q_idx"] in keep:
            p[r["q_idx"]][(r["i"], r["j"])] = r["p_con"]
    out = {}
    for item in json.load(open(cands_path))["data"]:
        q = item["q_idx"]
        if q not in keep:
            continue
        idx = sorted(c["p_idx"] for c in item["candidates"] if c["depth"] == 1)
        pairs = [max(p[q][(a, b)], p[q][(b, a)]) for k, a in enumerate(idx) for b in idx[k + 1:]]
        if pairs:
            out[q] = sum(pairs) / len(pairs)
    return out


def split_diff(rows, metric):
    """rows: list of (D, q, scores). Low half minus high half."""
    rows = sorted(rows, key=lambda r: (r[0], r[1]))
    h = len(rows) // 2
    low, high = rows[:h], rows[h:]
    m = lambda xs: sum(r[2][metric] for r in xs) / len(xs)
    return m(low), m(high), m(low) - m(high)


def auroc(rows):
    pos = [r[0] for r in rows if r[2]["str_em"] == 0]
    neg = [r[0] for r in rows if r[2]["str_em"] > 0]
    if not pos or not neg:
        return float("nan")
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True)
    ap.add_argument("--nli", required=True)
    ap.add_argument("--perq", required=True, help="per-question scores of the released choices (score_per_question.py)")
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n_boot", type=int, default=2000)
    a = ap.parse_args()

    keep = set(load_split_indices(a.split))
    D = disagreement(a.cands, a.nli, keep)
    scores = {r["q_idx"]: r for r in (json.loads(l) for l in open(a.perq) if l.strip()) if r["q_idx"] in keep}
    rows = [(D[q], q, scores[q]) for q in sorted(keep) if q in D and q in scores]
    print(f"questions: {len(rows)} (split {len(keep)}, with disagreement {len(D)}, with scores {len(scores)})")

    rng = random.Random(SEED)
    boots = [[rows[rng.randrange(len(rows))] for _ in rows] for _ in range(a.n_boot)]
    ci = lambda vals: (sorted(vals)[int(0.025 * len(vals))], sorted(vals)[int(0.975 * len(vals)) - 1])

    result = {"split": a.split, "n": len(rows), "median_D": sorted(r[0] for r in rows)[len(rows) // 2]}
    for metric in METRICS:
        low, high, diff = split_diff(rows, metric)
        lo, hi = ci([split_diff(b, metric)[2] for b in boots])
        result[metric] = {"low_disagreement": round(low, 2), "high_disagreement": round(high, 2),
                          "low_minus_high": round(diff, 2), "ci95": [round(lo, 2), round(hi, 2)],
                          "excludes_0": lo > 0 or hi < 0}
    au = auroc(rows)
    lo, hi = ci([auroc(b) for b in boots])
    result["auroc_D_for_chosen_str_em_0"] = {"value": round(au, 3), "ci95": [round(lo, 3), round(hi, 3)],
                                            "n_wrong": sum(r[2]["str_em"] == 0 for r in rows)}
    h = result["str_em"]
    result["H1_confirmed"] = h["ci95"][0] > 0
    print(json.dumps(result, indent=2))
    if a.out:
        json.dump(result, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
