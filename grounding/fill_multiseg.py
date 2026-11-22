"""Append per-question rows that lookup_perq.py could not score (multi-segment answers kept as released).

Such a question has no depth-1 candidate, so it is not in cand_metrics_*_exact.jsonl. Every re-ranker keeps its
released answer, so its scores are taken from the GPU scoring of the released file, after checking that the
system's output text is identical to the released one. See grounding/PREREG_C_deviations.md.

    python grounding/fill_multiseg.py --f out/rr_ground_C.json --perq out/rr_ground_C.json.perq.jsonl \
        --released results/runs/rr_released_C.json --released_perq results/runs/rr_released_C.json.perq.jsonl
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f", required=True, help="re-ranked file")
    ap.add_argument("--perq", required=True, help="its lookup .perq.jsonl (appended to in place)")
    ap.add_argument("--released", required=True)
    ap.add_argument("--released_perq", required=True)
    a = ap.parse_args()

    have = {json.loads(line)["q_idx"] for line in open(a.perq) if line.strip()}
    sys_out = {it["q_idx"]: it["output"] for it in json.load(open(a.f))["data"]}
    rel_out = {it["q_idx"]: it["output"] for it in json.load(open(a.released))["data"]}
    rel_perq = {r["q_idx"]: r for r in map(json.loads, open(a.released_perq))}
    added = []
    for q in sorted(set(sys_out) - have):
        assert sys_out[q] == rel_out[q], f"q{q}: output differs from released, cannot reuse its scores"
        added.append(rel_perq[q])
    with open(a.perq, "a") as f:
        for r in added:
            f.write(json.dumps(r) + "\n")
    print(f"appended {len(added)} rows {[r['q_idx'] for r in added]} -> {len(have) + len(added)} questions")


if __name__ == "__main__":
    main()
