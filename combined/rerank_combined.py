"""Combined system: re-pick each ASQA answer with every confirmed component switched on or off (CPU, offline).

Components (each was tested on its own against C1 before being combined; see combined/PREREG.md):
  passages   the candidates file decides it: GTR top 5 (results/runs/asqa_cands_*.json) or the retrieval
             division's cross-encoder reranked top 5 (results/runs/retrieval/asqa_rrR2K20*_0_end.json)
  --critique released  Self-RAG's ranking as the released code computes it (relevance + support)
  --critique paper     the paper's full critique score: relevance + support + 0.5 * utility + exp(sequence score)
                       (critique division, results/critique_paper_scoring.md)
  --ground ... --alpha grounding division's D2 term: + w_sup * alpha * (g - support), g = mean DeBERTa entailment
                       of the candidate's sentences by its own passage (grounding/cand_ground.py)
  --rel ... --delta    grounding division's D2-v2 relevance gate: only candidates whose passage relevance is at
                       most delta below the released choice's are eligible (grounding/qa_relevance.py)

The code of each component is imported from its division, not re-written:
  critique/learn_weights.py features()  and  grounding/rerank_ground.py load_grounding / load_relevance / gate / choose.
`--critique released` with no --ground and no --rel must reproduce the released choices exactly (checked, printed).
Multi-segment answers are kept as released, as in every division's re-ranker.

    python3 combined/rerank_combined.py --f results/runs/retrieval/asqa_rrR2K20C_0_end.json \
        --split critique/splits/asqa_C.json --critique paper \
        --ground results/runs/combined/cand_ground_rr_C.jsonl --alpha 0.75 \
        --rel results/runs/combined/cand_qarel_rr_C.jsonl --delta 1 --out out/combined/final_C.json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "critique"))
sys.path.insert(0, os.path.join(HERE, "..", "grounding"))
from common import load_split_indices  # noqa: E402
from rerank import build_output  # noqa: E402
from learn_weights import features  # noqa: E402
from rerank_ground import KEEP, choose, gate, load_grounding, load_relevance  # noqa: E402

# Paper / upstream README weights for the full critique score (critique division, no tuning)
PAPER = {"w_rel": 1.0, "w_sup": 1.0, "w_use": 0.5, "use_seqscore": True}


def critique_score(c, kind):
    if kind == "released":
        return c["final_score"]  # = relevance + support in the released code
    relevance, support, utility, seq = features(c, 1.0)
    return (PAPER["w_rel"] * relevance + PAPER["w_sup"] * support + PAPER["w_use"] * utility
            + (seq if PAPER["use_seqscore"] else 0.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--f", required=True, help="--save_candidates output file (decides the passages)")
    ap.add_argument("--split", required=True, help="comma-separated split files")
    ap.add_argument("--critique", choices=["released", "paper"], default="released")
    ap.add_argument("--ground", default="", help="comma-separated cand_ground_*.jsonl (enables the D2 term)")
    ap.add_argument("--agg", default="mean")
    ap.add_argument("--alpha", type=float, default=0.75, help="D2 mix weight (grounding division: 0.75)")
    ap.add_argument("--rel", default="", help="comma-separated cand_qarel_*.jsonl (enables the D2-v2 gate)")
    ap.add_argument("--rel_model", default="BAAI/bge-reranker-base")
    ap.add_argument("--delta", type=float, default=1.0, help="gate margin (grounding division: 1)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    full = json.load(open(a.f))
    w_sup = float(full.get("args", {}).get("w_sup", 1.0))
    g = load_grounding(a.ground, a.agg) if a.ground else None
    rel = load_relevance(a.rel, a.rel_model) if a.rel else None
    keep_q = set(load_split_indices(a.split))

    data, n_multi, n_changed, n_missing_g = [], 0, 0, 0
    for item in full["data"]:
        if item["q_idx"] not in keep_q:
            continue
        path = item["best_paths"][str(item.get("output_path", 0))]
        depth1 = sorted([c for c in item["candidates"] if c["depth"] == 1], key=lambda c: c["p_idx"])
        if len([n for n in path if n not in (None, 0)]) > 1 or not depth1:
            n_multi += 1
            data.append({k: v for k, v in item.items() if k not in ("candidates", "best_paths")})
            continue
        scores = {}
        for c in depth1:
            s = critique_score(c, a.critique)
            if g is not None:
                gc = g.get((item["q_idx"], c["node_id"]))
                if gc is None:
                    n_missing_g += 1
                    gc = 0.0
                s += w_sup * a.alpha * (gc - c["ground_score"])
            scores[c["node_id"]] = s
        pool = gate(depth1, path[-1], item["q_idx"], rel, a.delta) if rel is not None else depth1
        chosen = choose(pool, scores)
        new_item = {k: item[k] for k in KEEP if k in item}
        new_item.update({"output": build_output(item, chosen), "docs": [chosen["ctx"]],
                         "intermediate": ["[Retrieval]", chosen["pred"]], "chosen_p_idx": chosen["p_idx"],
                         "chosen_node_id": chosen["node_id"],
                         "original_p_idx": next(c["p_idx"] for c in depth1 if c["node_id"] == path[-1])})
        n_changed += chosen["node_id"] != path[-1]
        data.append(new_item)

    config = {"source": a.f, "split": a.split, "critique": a.critique, "paper_weights": PAPER if a.critique == "paper"
              else None, "ground": a.ground or None, "agg": a.agg if a.ground else None,
              "alpha": a.alpha if a.ground else None, "rel": a.rel or None,
              "rel_model": a.rel_model if a.rel else None, "delta": a.delta if a.rel else None, "w_sup": w_sup}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"data": data, "args": config}, open(a.out, "w"))
    json.dump(config, open(a.out + ".config.json", "w"), indent=2)
    print(f"{len(data)} questions, {n_multi} multi-segment kept, answer changed vs this file's released choice: "
          f"{n_changed}, candidates without a grounding score: {n_missing_g}")
    if a.critique == "released" and g is None and rel is None and n_changed:
        sys.exit("ERROR: the released setting must reproduce every released choice")
    print("RERANK_FILE", a.out)


if __name__ == "__main__":
    main()
