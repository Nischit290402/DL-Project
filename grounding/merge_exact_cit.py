"""Add exact ALCE citation recall/precision (from the per-slot T5-11B runs) to cand_metrics_*.jsonl.

Input: cand_metrics file + the <prefix>_p{k}.json.perq.jsonl files written by score_per_question.py.
Output: a copy of the cand_metrics rows with citation_rec / citation_prec added (None where ALCE gives none).
Checks: str-em and ROUGE-Lsum per candidate must equal cand_metrics (same answer text), and the selected
candidates must reproduce the released per-question citation (if --released is given).

    python grounding/merge_exact_cit.py --metrics results/runs/cand_metrics_AB.jsonl --perq_prefix out/pick_AB \
        --released results/runs/rr_released.json.perq.jsonl --out results/runs/cand_metrics_AB_exact.jsonl
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--perq_prefix", required=True)
    ap.add_argument("--slots", type=int, default=5)
    ap.add_argument("--released", default="", help="released per-question scores, for the cross-check")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    exact = {}
    for k in range(a.slots):
        path = f"{a.perq_prefix}_p{k}.json.perq.jsonl"
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}: did the scoring job finish?")
        for line in open(path):
            r = json.loads(line)
            exact[(r["q_idx"], k)] = r

    rows = [json.loads(line) for line in open(a.metrics)]
    n_found, mismatch = 0, 0
    for row in rows:
        r = exact.get((row["q_idx"], row["p_idx"]))
        row["citation_rec"] = r.get("citation_rec") if r else None
        row["citation_prec"] = r.get("citation_prec") if r else None
        if r:
            n_found += 1
            if abs(r["str_em"] - row["str_em"]) > 1e-6 or abs(r["rougeLsum"] - row["rougeLsum"]) > 1e-6:
                mismatch += 1
    with open(a.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"{n_found} of {len(rows)} candidates have exact citation; str-em/ROUGE mismatches: {mismatch}")

    if a.released:
        rel = {json.loads(l)["q_idx"]: json.loads(l) for l in open(a.released)}
        same, n = 0, 0
        for row in rows:
            if row["selected"] and row["q_idx"] in rel and row["citation_rec"] is not None:
                n += 1
                same += abs(rel[row["q_idx"]].get("citation_rec", -1) - row["citation_rec"]) < 1e-6
        print(f"released cross-check: citation_rec identical on {same} of {n} selected candidates")
    print("EXACT_FILE", a.out)


if __name__ == "__main__":
    main()
