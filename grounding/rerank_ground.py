"""D2: re-pick each ASQA answer with the DeBERTa grounding score added to Self-RAG's ranking (CPU, offline).

Same machinery as critique/rerank.py (one-segment answers are decided by one choice among the depth-1 candidates,
so re-picking offline equals what Self-RAG would output). Only the ranking score changes:

  --mode add   score = released + mu * g
  --mode mix   score = released - w_sup * ISSUP + w_sup * ((1 - alpha) * ISSUP + alpha * g)
               (= released + w_sup * alpha * (g - ISSUP))
  g = DeBERTa grounding of the candidate (grounding/cand_ground.py), aggregated over its ALCE sentences:
  --agg mean       mean p_ent           --agg soft_mean  mean of p_ent + 0.5 * p_neu
  --agg min / first / full              (see grounding/diagnostics.py)

--param 0 must reproduce the released choices exactly (checked and printed).
Writes an ALCE-format file to score with cluster/run_critique_score.sbatch, plus <out>.config.json.

    python grounding/rerank_ground.py --f results/runs/asqa_cands_0_end.json --ground results/runs/cand_ground_A.jsonl \
        --split critique/splits/asqa_A.json --mode add --agg mean --param 0.5 --out out/rr_ground_A.json
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "critique"))
sys.path.insert(0, HERE)
from common import load_split_indices, postprocess  # noqa: E402
from rerank import build_output  # noqa: E402
from diagnostics import aggregates  # noqa: E402

KEEP = ("question", "qa_pairs", "annotations", "answer", "sample_id", "q_idx")


def load_grounding(paths, agg):
    rows = collections.defaultdict(list)
    for path in [p for p in paths.split(",") if p]:
        for line in open(path):
            r = json.loads(line)
            rows[(r["q_idx"], r["node_id"])].append(r)
    return {k: aggregates(v)[agg] for k, v in rows.items()}


def new_score(c, g, mode, param, w_sup):
    if mode == "add":
        return c["final_score"] + param * g
    if mode == "mix":
        return c["final_score"] + w_sup * param * (g - c["ground_score"])
    raise ValueError(mode)


def choose(depth1, scores):
    """Highest score, stable on ties (passage order); same empty-answer fallback as run_long_form_static.py."""
    ranked = sorted(depth1, key=lambda c: scores[c["node_id"]], reverse=True)
    chosen = ranked[0]
    if len(postprocess(" " + chosen["processed_pred"])) == 0 and len(ranked) > 1:
        chosen = ranked[1]
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f", required=True, help="--save_candidates output file")
    ap.add_argument("--ground", required=True, help="comma-separated cand_ground_*.jsonl files")
    ap.add_argument("--split", required=True, help="comma-separated split files")
    ap.add_argument("--mode", required=True, choices=["add", "mix"])
    ap.add_argument("--agg", default="mean", choices=["mean", "soft_mean", "min", "first", "full"])
    ap.add_argument("--param", type=float, required=True, help="mu (add) or alpha (mix)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    full = json.load(open(a.f))
    w_sup = float(full.get("args", {}).get("w_sup", 1.0))
    g = load_grounding(a.ground, a.agg)
    keep_q = set(load_split_indices(a.split))
    data, n_multi, n_changed, n_missing = [], 0, 0, 0
    for item in full["data"]:
        if item["q_idx"] not in keep_q:
            continue
        path = item["best_paths"][str(item.get("output_path", 0))]
        depth1 = sorted([c for c in item["candidates"] if c["depth"] == 1], key=lambda c: c["p_idx"])
        if len([n for n in path if n not in (None, 0)]) > 1 or not depth1:
            n_multi += 1  # multi-segment: keep the original answer, as critique/rerank.py does
            data.append({k: v for k, v in item.items() if k not in ("candidates", "best_paths")})
            continue
        scores = {}
        for c in depth1:
            gc = g.get((item["q_idx"], c["node_id"]))
            if gc is None:
                n_missing += 1
                gc = 0.0
            scores[c["node_id"]] = new_score(c, gc, a.mode, a.param, w_sup)
        chosen = choose(depth1, scores)
        new_item = {k: item[k] for k in KEEP if k in item}
        new_item.update({"output": build_output(item, chosen), "docs": [chosen["ctx"]],
                         "intermediate": ["[Retrieval]", chosen["pred"]], "chosen_p_idx": chosen["p_idx"],
                         "chosen_node_id": chosen["node_id"],
                         "original_p_idx": next(c["p_idx"] for c in depth1 if c["node_id"] == path[-1])})
        n_changed += chosen["node_id"] != path[-1]
        data.append(new_item)

    config = {"source": a.f, "ground": a.ground, "split": a.split, "mode": a.mode, "agg": a.agg,
              "param": a.param, "w_sup": w_sup, "model": "cross-encoder/nli-deberta-v3-large"}
    json.dump({"data": data, "args": config}, open(a.out, "w"))
    json.dump(config, open(a.out + ".config.json", "w"), indent=2)
    print(json.dumps(config))
    print(f"{len(data)} questions, {n_multi} multi-segment kept, answer changed vs released: {n_changed}, "
          f"candidates without a grounding score: {n_missing}")
    if a.param == 0 and n_changed:
        print("WARNING: param 0 should reproduce every released choice")
    print("RERANK_FILE", a.out)


if __name__ == "__main__":
    main()
