"""D2 Step 3: tune the grounding term on Set A by table lookup (CPU). Refuses Set B / C.

Every setting re-picks each question's answer (same chooser as grounding/rerank_ground.py) and looks up the
chosen candidate's exact ALCE metrics in cand_metrics_AB_exact.jsonl, so no GPU is needed.

Grid (fixed before running):
  add  mu    in {0.1, 0.25, 0.5, 0.75, 1.0}
  mix  alpha in {0.25, 0.5, 0.75, 1.0}
  aggregation `mean` (chosen in Step 2); `soft_mean` rows are reported for robustness only, never selected.

Selection rule (fixed before running, written down in results/nli_grounding_tune_A.md):
  1. Eligible: Set A str-em difference vs released >= -0.5 (the team's noise threshold).
  2. c* = best citation-recall gain among eligible `mean` settings; plateau = eligible `mean` settings with
     gain >= c* - 1.0.
  3. Mode = the mode with more settings on the plateau (tie -> add, the simpler one); parameter = the median of
     that mode's plateau parameters (lower median if even), i.e. the middle of the plateau, not the peak.
  4. If nothing is eligible: the setting with the smallest str-em loss among those with a citation gain > 0.

    python grounding/tune_ground.py --out results/ground_tune_A
"""
import argparse
import collections
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "critique"))
sys.path.insert(0, HERE)
from common import load_split_indices  # noqa: E402
from diagnostics import aggregates  # noqa: E402

SEED = 6103
REPS = 2000
GRID = [("add", p) for p in (0.1, 0.25, 0.5, 0.75, 1.0)] + [("mix", p) for p in (0.25, 0.5, 0.75, 1.0)]
METRICS = ("str_em", "citation_rec", "citation_prec", "rougeLsum", "length")


def score(r, mode, p, w_sup=1.0):
    return r["final_score"] + (p * r["g"] if mode == "add" else w_sup * p * (r["g"] - r["ground_score"]))


def pick(cands, key):
    ranked = sorted(cands, key=key, reverse=True)
    return ranked[1] if ranked[0]["empty"] and len(ranked) > 1 else ranked[0]


def paired_ci(base, new, rng):
    d = [n - b for b, n in zip(base, new) if b is not None and n is not None]
    boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(REPS))
    return [round(sum(d) / len(d), 2), round(boots[int(0.025 * REPS)], 2), round(boots[int(0.975 * REPS)], 2)]


def evaluate(qs, chooser, base):
    rng = random.Random(SEED)
    chosen = [chooser(c) for c in qs]
    out = {"changed": sum(x["node_id"] != b["node_id"] for x, b in zip(chosen, base)),
           "choices": {str(c[0]["q_idx"]): x["node_id"] for c, x in zip(qs, chosen)}}
    for m in METRICS:
        vals = [x[m] for x in chosen if x[m] is not None]
        out[m] = round(sum(vals) / len(vals), 2)
        out["d_" + m] = paired_ci([b[m] for b in base], [x[m] for x in chosen], rng)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="critique/splits/asqa_A.json")
    ap.add_argument("--ground", default="results/runs/cand_ground_A.jsonl")
    ap.add_argument("--metrics", default="results/runs/cand_metrics_AB_exact.jsonl")
    ap.add_argument("--out", default="results/ground_tune_A")
    a = ap.parse_args()
    assert a.split.endswith("asqa_A.json"), "tuning is on Set A only"

    split = set(load_split_indices(a.split))
    grows = collections.defaultdict(list)
    for line in open(a.ground):
        r = json.loads(line)
        grows[(r["q_idx"], r["node_id"])].append(r)
    by_q = collections.defaultdict(list)
    for line in open(a.metrics):
        r = json.loads(line)
        if r["q_idx"] in split and not r["multi_segment"]:
            r["agg"] = aggregates(grows[(r["q_idx"], r["node_id"])])
            by_q[r["q_idx"]].append(r)
    qs = [sorted(v, key=lambda r: r["p_idx"]) for _, v in sorted(by_q.items())]
    base = [next(r for r in c if r["selected"]) for c in qs]
    assert all(pick(c, lambda r: r["final_score"])["node_id"] == b["node_id"] for c, b in zip(qs, base))

    rows = [{"mode": "released", "param": 0, "agg": "-", **evaluate(qs, lambda c: next(r for r in c if r["selected"]), base)}]
    for agg in ("mean", "soft_mean"):
        for c in qs:
            for r in c:
                r["g"] = r["agg"][agg]
        for mode, p in GRID:
            rows.append({"mode": mode, "param": p, "agg": agg,
                         **evaluate(qs, lambda c, mode=mode, p=p: pick(c, lambda r: score(r, mode, p)), base)})

    # selection rule (see docstring)
    cands = [r for r in rows if r["agg"] == "mean"]
    eligible = [r for r in cands if r["d_str_em"][0] >= -0.5]
    if eligible:
        best = max(r["d_citation_rec"][0] for r in eligible)
        plateau = [r for r in eligible if r["d_citation_rec"][0] >= best - 1.0]
        counts = collections.Counter(r["mode"] for r in plateau)
        mode = "add" if counts["add"] >= counts["mix"] else "mix"
        ps = sorted(r["param"] for r in plateau if r["mode"] == mode)
        param = ps[(len(ps) - 1) // 2]
        rule = {"eligible": len(eligible), "best_citation_gain": best,
                "plateau": [(r["mode"], r["param"]) for r in plateau], "mode": mode, "param": param}
    else:
        pos = [r for r in cands if r["d_citation_rec"][0] > 0]
        pick_row = max(pos, key=lambda r: r["d_str_em"][0])
        rule = {"eligible": 0, "fallback": True, "mode": pick_row["mode"], "param": pick_row["param"]}
    chosen = next(r for r in cands if r["mode"] == rule["mode"] and r["param"] == rule["param"])
    rule["chosen_metrics"] = {k: chosen[k] for k in chosen if k.startswith("d_") or k == "changed"}

    json.dump({"rows": rows, "selection": rule}, open(a.out + ".json", "w"), indent=1)
    print(f"{'agg':10s}{'mode':9s}{'param':>6s}{'chg':>5s}  {'str-em':>22s}  {'cit rec':>22s}  {'cit prec':>22s}  {'rouge':>7s} {'len':>5s}")
    for r in rows:
        print(f"{r['agg']:10s}{r['mode']:9s}{r['param']:>6}{r['changed']:>5}  {str(r['d_str_em']):>22s}  "
              f"{str(r['d_citation_rec']):>22s}  {str(r['d_citation_prec']):>22s}  {r['d_rougeLsum'][0]:>7} {r['length']:>5}")
    print("SELECTION", json.dumps({k: v for k, v in rule.items() if k != "chosen_metrics"}))


if __name__ == "__main__":
    main()
