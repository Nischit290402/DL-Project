"""D2-v2: tune the relevance-gated grounding re-ranker on Set A+B by table lookup (CPU). Refuses Set C.

Setting, grid and selection rule are fixed in results/nli_grounding_v2_rule.md (written before this ran; its
SHA-256 is stored in the output so later edits are visible). Same chooser as grounding/rerank_ground.py --rel.

    python grounding/tune_ground_v2.py --out results/ground_tune_v2_AB
"""
import argparse
import collections
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "critique"))
sys.path.insert(0, HERE)
from common import load_split_indices  # noqa: E402
from diagnostics import aggregates  # noqa: E402
from tune_ground import evaluate, pick  # noqa: E402

ALPHAS = (0.5, 0.75, 1.0)
DELTAS = (0.0, 1.0, 2.0)
RULE = os.path.join(HERE, "..", "results", "nli_grounding_v2_rule.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="critique/splits/asqa_A.json,critique/splits/asqa_B.json")
    ap.add_argument("--ground", default="results/runs/cand_ground_A.jsonl,results/runs/cand_ground_B.jsonl")
    ap.add_argument("--rel", default="results/runs/cand_qarel_AB.jsonl")
    ap.add_argument("--rel_model", default="BAAI/bge-reranker-base")
    ap.add_argument("--metrics", default="results/runs/cand_metrics_AB_exact.jsonl")
    ap.add_argument("--out", default="results/ground_tune_v2_AB")
    a = ap.parse_args()
    assert "asqa_C" not in a.split and "_C" not in a.ground + a.rel, "D2-v2 is tuned on A+B only"

    split = set()
    for s in a.split.split(","):
        split |= set(load_split_indices(s))
    grows = collections.defaultdict(list)
    for path in a.ground.split(","):
        for line in open(path):
            r = json.loads(line)
            grows[(r["q_idx"], r["node_id"])].append(r)
    rel = {}
    for line in open(a.rel):
        r = json.loads(line)
        if r["model"] == a.rel_model:
            rel[(r["q_idx"], r["node_id"])] = r["score"]
    by_q = collections.defaultdict(list)
    for line in open(a.metrics):
        r = json.loads(line)
        if r["q_idx"] in split and not r["multi_segment"]:
            r["g"] = aggregates(grows[(r["q_idx"], r["node_id"])])["mean"]
            r["rel"] = rel[(r["q_idx"], r["node_id"])]
            by_q[r["q_idx"]].append(r)
    qs = [sorted(v, key=lambda r: r["p_idx"]) for _, v in sorted(by_q.items())]
    base = [next(r for r in c if r["selected"]) for c in qs]
    print(f"{len(qs)} single-segment questions in A+B")

    def chooser(alpha, delta):
        def f(c):
            r0 = next(r for r in c if r["selected"])
            pool = c if delta is None else [r for r in c if r["selected"] or r["rel"] >= r0["rel"] - delta]
            return pick(pool, lambda r: r["final_score"] + alpha * (r["g"] - r["ground_score"]))
        return f

    rows = [{"alpha": 0, "delta": None, "ref": True, **evaluate(qs, chooser(0.0, None), base)}]
    rows += [{"alpha": al, "delta": None, "ref": True, **evaluate(qs, chooser(al, None), base)} for al in ALPHAS]
    rows += [{"alpha": al, "delta": de, "ref": False, **evaluate(qs, chooser(al, de), base)}
             for al in ALPHAS for de in DELTAS]
    assert rows[0]["changed"] == 0, "alpha 0 must reproduce the released choices"

    grid = [r for r in rows if not r["ref"]]
    eligible = [r for r in grid if r["d_str_em"][0] >= -0.5 and r["d_rougeLsum"][0] >= -0.5]
    if eligible:
        best = max(r["d_citation_rec"][0] for r in eligible)
        plateau = [r for r in eligible if r["d_citation_rec"][0] >= best - 1.0]
        dist = lambda r: abs(ALPHAS.index(r["alpha"]) - 1) + abs(DELTAS.index(r["delta"]) - 1)
        chosen = min(plateau, key=lambda r: (dist(r), r["delta"], r["alpha"]))
        sel = {"eligible": len(eligible), "best_citation_gain": best,
               "plateau": [(r["alpha"], r["delta"]) for r in plateau]}
    else:
        fb = [r for r in grid if r["d_str_em"][0] >= -0.5 and r["d_rougeLsum"][0] >= -1.0]
        chosen = max(fb, key=lambda r: r["d_citation_rec"][0]) if fb else None
        sel = {"eligible": 0, "fallback": True, "dropped": chosen is None}
    if chosen:
        sel.update({"alpha": chosen["alpha"], "delta": chosen["delta"],
                    "chosen_metrics": {k: chosen[k] for k in chosen if k.startswith("d_") or k == "changed"}})
    sel["rule_sha256"] = hashlib.sha256(open(RULE, "rb").read()).hexdigest()

    json.dump({"rows": rows, "selection": sel}, open(a.out + ".json", "w"), indent=1)
    print(f"{'alpha':>6s}{'delta':>7s}{'chg':>5s}  {'str-em':>22s}  {'ROUGE-L':>22s}  {'cit rec':>22s}  {'len':>5s}")
    for r in rows:
        tag = " (ref, ungated)" if r["ref"] and r["alpha"] else (" (released)" if r["ref"] else "")
        print(f"{r['alpha']:>6}{str(r['delta']):>7s}{r['changed']:>5}  {str(r['d_str_em']):>22s}  "
              f"{str(r['d_rougeLsum']):>22s}  {str(r['d_citation_rec']):>22s}  {r['length']:>5}{tag}")
    print("SELECTION", json.dumps({k: v for k, v in sel.items() if k != "chosen_metrics"}))


if __name__ == "__main__":
    main()
