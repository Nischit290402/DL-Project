"""Re-pick each question's answer with a penalty for being contradicted by the other candidates (laptop or job, stdlib).

new score_i = final_score_i - LAMBDA * C_i,
C_i = mean over the other depth-1 candidates j of max(p_con(i->j), p_con(j->i))   (variant noctx).

Fixed on Set A on 2026-11-17, BEFORE Set B was run (see contradiction/README.md, Step 2):
  variant = noctx, LAMBDA = 0.3. Set A str-em 27.78 -> 28.50 (+0.72 [-2.00, +3.25]), 21 of 100 answers changed.
Do not change these defaults after looking at Set B.

Everything else matches critique/rerank.py: same output format, same fallback when the chosen candidate is empty,
multi-segment questions copied unchanged. With --lam 0 it must reproduce the released choices.

    python3 contradiction/rerank_contra.py --f results/runs/asqa_cands_0_end.json \
        --nli results/runs/cand_nli_B.jsonl --split critique/splits/asqa_B.json --out results/runs/rr_contra_B.json
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "critique"))
from common import load_split_indices, postprocess  # noqa: E402
from rerank import build_output  # noqa: E402

LAMBDA = 0.3
VARIANT = "noctx"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f", required=True, help="output file made with --save_candidates")
    ap.add_argument("--nli", required=True, help="cand_nli_<SET>.jsonl from contradiction/cand_nli.py")
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lam", type=float, default=LAMBDA)
    ap.add_argument("--variant", default=VARIANT, choices=["noctx", "ctx"])
    a = ap.parse_args()

    keep = set(load_split_indices(a.split))
    table = collections.defaultdict(dict)
    for line in open(a.nli):
        r = json.loads(line)
        if r["variant"] == a.variant:
            table[r["q_idx"]][(r["i"], r["j"])] = r["p_con"]

    fields = ("question", "qa_pairs", "annotations", "answer", "sample_id", "q_idx")
    data, n_multi, n_changed = [], 0, 0
    for item in json.load(open(a.f))["data"]:
        q = item["q_idx"]
        if q not in keep:
            continue
        path = item["best_paths"][str(item.get("output_path", 0))]
        original_node = path[-1]
        depth1 = sorted([c for c in item["candidates"] if c["depth"] == 1], key=lambda c: c["p_idx"])
        if len([n for n in path if n not in (None, 0)]) > 1 or not depth1:
            n_multi += 1
            data.append({k: v for k, v in item.items() if k not in ("candidates", "best_paths")})
            continue
        assert q in table, f"no NLI scores for question {q}"
        t = table[q]
        for c in depth1:
            others = [d for d in depth1 if d is not c]
            c["C"] = sum(max(t[(c["p_idx"], d["p_idx"])], t[(d["p_idx"], c["p_idx"])]) for d in others) / len(others)
            c["new_score"] = c["final_score"] - a.lam * c["C"]
        ranked = sorted(depth1, key=lambda c: c["new_score"], reverse=True)  # stable: ties keep passage order
        chosen = ranked[0]
        if len(postprocess(" " + chosen["processed_pred"])) == 0 and len(ranked) > 1:
            chosen = ranked[1]
        new_item = {k: item[k] for k in fields if k in item}
        new_item.update({"output": build_output(item, chosen), "docs": [chosen["ctx"]],
                         "intermediate": ["[Retrieval]", chosen["pred"]], "chosen_p_idx": chosen["p_idx"],
                         "chosen_C": round(chosen["C"], 4),
                         "original_p_idx": next(c["p_idx"] for c in depth1 if c["node_id"] == original_node)})
        n_changed += chosen["node_id"] != original_node
        data.append(new_item)

    settings = {"source": a.f, "nli": a.nli, "split": a.split, "variant": a.variant, "lambda": a.lam}
    json.dump({"data": data, "args": settings}, open(a.out, "w"))
    print(json.dumps(settings))
    print(f"{len(data)} questions, {n_multi} multi-segment kept as is, answer changed: {n_changed}")
    if a.lam == 0 and n_changed:
        print("WARNING: lambda 0 should reproduce every original choice")
    print("RERANK_FILE", a.out)


if __name__ == "__main__":
    main()
