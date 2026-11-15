"""Re-pick each question's answer from the saved candidates with a changed support grade. CPU only, no Self-RAG run.

Input: an output file made with --save_candidates. Every other part of the ranking is kept exactly as
run_long_form_static.py computes it; only the support term (w_sup * ground_score) is swapped:

  --variant first   the first grade, as released (with --temperature 1 this must reproduce the original choices)
  --variant mean    average support score over every grade in the candidate          (Fix 2)
  --variant min     lowest support score over every grade in the candidate           (Fix 2)
  --temperature T   temperature-scale the grade probabilities first                   (Fix 1, T from honesty_check.py)

Why re-picking offline gives exactly what Self-RAG would output: on ASQA, 947 of 948 answers are one segment, so
the answer is decided by one choice among the depth-1 candidates. Questions whose original answer has more than
one segment are copied unchanged and counted.

Writes an ALCE-format file to score with cluster/run_critique_score.sbatch.
    python critique/rerank.py --f out/asqa_cands_0_end.json --variant first --out out/rerank_first.json
    python critique/rerank.py --f out/asqa_cands_0_end.json --variant first --temperature 2.5 --out out/rerank_fix1.json
    python critique/rerank.py --f out/asqa_cands_0_end.json --variant mean --out out/rerank_fix2_mean.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import fix_spacing, postprocess  # noqa: E402
from honesty_check import support_score  # noqa: E402


def new_ground(cand, variant, temperature):
    scores = [support_score(p, temperature) for p in cand["support_probs"]]
    if not scores:
        return 0.0  # no grade written: the released code also uses 0
    if variant == "first":
        return scores[0]
    if variant == "mean":
        return sum(scores) / len(scores)
    if variant == "min":
        return min(scores)
    raise ValueError(variant)


def build_output(item, cand):
    """Same postprocessing as run_long_form_static.py main() for a one-segment answer."""
    sentences = ["", cand["processed_pred"]]  # root placeholder + the chosen segment
    final_output, prev_gen = "", []
    for idx, sent in enumerate(sentences):
        if len(sent) == 0:
            continue
        pp = postprocess(sent)
        if pp in prev_gen:
            continue
        prev_gen.append(pp)
        final_output += pp[:-1] + " [{}]".format(idx) + ". "
    if len(final_output) > 0 and final_output[-1] == " ":
        final_output = final_output[:-1]
    output = fix_spacing(final_output)
    output = output.replace(".[Continue to Use Evidence]", " [1]. ")
    return output.replace(". [1] ", " [1]. ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", required=True, help="output file made with --save_candidates")
    parser.add_argument("--variant", default="first", choices=["first", "mean", "min"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    full = json.load(open(args.f))
    w_sup = float(full.get("args", {}).get("w_sup", 1.0))
    keep = ("question", "qa_pairs", "annotations", "answer", "sample_id", "q_idx")
    data, n_multi, n_changed, n_same_text = [], 0, 0, 0
    n_cands, n_cands_multi_grade = 0, 0

    for item in full["data"]:
        path = item["best_paths"][str(item.get("output_path", 0))]
        original_node = path[-1]
        depth1 = [c for c in item["candidates"] if c["depth"] == 1]
        if len(path) > 2 or not depth1:  # multi-segment (or no candidates): keep the original answer
            n_multi += 1
            data.append({k: v for k, v in item.items() if k not in ("candidates", "best_paths")})
            continue
        n_cands += len(depth1)
        n_cands_multi_grade += sum(len(c["support_probs"]) > 1 for c in depth1)
        for c in depth1:
            c["new_score"] = c["final_score"] - w_sup * c["ground_score"] + w_sup * new_ground(
                c, args.variant, args.temperature)
        ranked = sorted(depth1, key=lambda c: c["new_score"], reverse=True)  # stable: ties keep passage order
        chosen = ranked[0]
        if len(postprocess(" " + chosen["processed_pred"])) == 0 and len(ranked) > 1:
            chosen = ranked[1]  # same fallback as main(): the best path is empty after postprocessing
        new_item = {k: item[k] for k in keep if k in item}
        new_item.update({"output": build_output(item, chosen), "docs": [chosen["ctx"]],
                         "intermediate": ["[Retrieval]", chosen["pred"]], "chosen_p_idx": chosen["p_idx"],
                         "original_p_idx": next(c["p_idx"] for c in depth1 if c["node_id"] == original_node)})
        n_changed += chosen["node_id"] != original_node
        n_same_text += new_item["output"] == item["output"]
        data.append(new_item)

    settings = {"source": args.f, "variant": args.variant, "temperature": args.temperature, "w_sup": w_sup}
    json.dump({"data": data, "args": settings}, open(args.out, "w"))
    n_single = len(data) - n_multi
    print(json.dumps(settings))
    print(f"{len(data)} questions: {n_single} re-picked, {n_multi} multi-segment kept as is")
    print(f"answer changed vs the original run: {n_changed} of {n_single}")
    print(f"output text identical to the original run: {n_same_text} of {n_single}")
    print(f"candidates with more than one support grade (where Fix 2 can differ): {n_cands_multi_grade} of {n_cands}")
    if args.variant == "first" and args.temperature == 1.0 and n_changed:
        print("WARNING: the released setting should reproduce every original choice; check before trusting results")
    print("RERANK_FILE", args.out)


if __name__ == "__main__":
    main()
