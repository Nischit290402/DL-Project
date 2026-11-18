"""Write one ALCE-format file per passage slot k (0..4): every question answers with its k-th depth-1 candidate.

Scoring these files with critique/score_per_question.py (T5-11B) gives the exact ALCE citation recall/precision of
every candidate, which cand_metrics.py can only approximate (cit_proxy). CPU only, a few seconds; run inside the GPU
job (cluster/run_cand_exact_cit.sbatch), not on the login node.

Multi-segment questions and questions without a k-th candidate are left out of file k (merge_exact_cit.py leaves
their value empty). The answer text is built exactly as in critique/rerank.py, so str-em / ROUGE of file k match
cand_metrics_*.jsonl (merge_exact_cit.py checks this).

    python grounding/make_pick_files.py --f results/runs/asqa_cands_0_end.json --out_prefix out/pick_AB
    -> out/pick_AB_p0.json ... out/pick_AB_p4.json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "critique"))
from rerank import build_output  # noqa: E402

KEEP = ("question", "qa_pairs", "annotations", "answer", "sample_id", "q_idx")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f", required=True, help="--save_candidates output file")
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--slots", type=int, default=5)
    a = ap.parse_args()

    data = json.load(open(a.f))["data"]
    os.makedirs(os.path.dirname(a.out_prefix) or ".", exist_ok=True)
    for k in range(a.slots):
        out, skipped = [], 0
        for item in data:
            path = item["best_paths"][str(item.get("output_path", 0))]
            n_segments = len([x for x in path if x not in (None, 0)])
            cand = [c for c in item["candidates"] if c["depth"] == 1 and c["p_idx"] == k]
            if n_segments > 1 or not cand:
                skipped += 1
                continue
            c = cand[0]
            new_item = {key: item[key] for key in KEEP if key in item}
            new_item.update({"output": build_output(item, c), "docs": [c["ctx"]],
                             "intermediate": ["[Retrieval]", c["pred"]], "p_idx": k, "node_id": c["node_id"]})
            out.append(new_item)
        path_k = f"{a.out_prefix}_p{k}.json"
        json.dump({"data": out, "args": {"source": a.f, "slot": k}}, open(path_k, "w"))
        print(f"slot {k}: {len(out)} questions, {skipped} skipped -> {path_k}")


if __name__ == "__main__":
    main()
