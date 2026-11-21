"""Per-question ALCE scores of a re-ranked file by lookup in cand_metrics_*_exact.jsonl (CPU, seconds).

Every candidate's exact ALCE metrics (str-em, ROUGE-Lsum, length, citation rec/prec with T5-11B) were computed
once (cluster/run_cand_exact_cit.sbatch). A re-ranked file only chooses among those candidates, so its
per-question scores are the chosen candidates' rows: identical to running critique/score_per_question.py on it.
Writes <f>.perq.jsonl in the same format, ready for critique/bootstrap.py.

    python grounding/lookup_perq.py --f out/rr_ground_B.json --metrics results/runs/cand_metrics_AB_exact.jsonl
"""
import argparse
import json

KEYS = ("length", "str_em", "rougeLsum", "citation_rec", "citation_prec")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f", required=True, help="re-ranked file from grounding/rerank_ground.py")
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    by_cand = {}
    for line in open(a.metrics):
        r = json.loads(line)
        by_cand[(r["q_idx"], r["node_id"])] = r
    rows, missing = [], []
    for it in json.load(open(a.f))["data"]:
        r = by_cand.get((it["q_idx"], it.get("chosen_node_id")))
        if r is None:
            missing.append(it["q_idx"])  # multi-segment answer kept as is: not in the candidate table
            continue
        rows.append({"q_idx": it["q_idx"], **{k: r[k] for k in KEYS}})
    out = a.out or a.f + ".perq.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} questions scored by lookup, {len(missing)} not in the table {missing[:5]}")
    print("PERQ_FILE", out)


if __name__ == "__main__":
    main()
