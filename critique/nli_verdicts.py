"""Outside-checker verdicts for Self-RAG's own support grades (the honesty check, GPU step).

For every [IsSup] grade the model wrote ([Fully supported] / [Partially supported] / [No support / Contradictory]),
take the text it graded (the words since the previous grade) and the passage the candidate was written from, and
ask the same NLI model ALCE uses for citation recall (google/t5_xxl_true_nli_mixture, ALCE's exact prompt):
verdict 1 = the passage supports the text, 0 = it does not.

Two input kinds:
  * a --save_candidates output file: all candidates of every question, with the model's grade probabilities
    (full analysis: honesty_check.py can then measure calibration and fit Fix 1);
  * a normal output file such as C1 (out/asqa_keeptok_0_end.json): only the chosen answer and its grade labels,
    no probabilities (label-only check, works without any new Self-RAG run).

Writes one JSON line per graded claim.
GPU job: sbatch --export=ALL,F=out/asqa_cands_0_end.json cluster/run_critique_nli.sbatch
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SUPPORT_TOKENS, clean_claim, load_outputs, load_split_indices  # noqa: E402

SUPPORT_RE = re.compile("|".join(re.escape(t) for t in SUPPORT_TOKENS))


def graded_claims(pred):
    """[(claim text, label)] for each support grade in the generated text, in order."""
    claims, start = [], 0
    for m in SUPPORT_RE.finditer(pred):
        claims.append((clean_claim(pred[start:m.start()]), m.group(0)))
        start = m.end()
    return claims


def candidate_records(item):
    """Candidates of one question as dicts with pred, ctx, probabilities (None if not saved) and flags."""
    if "candidates" in item:
        path = item["best_paths"][str(item.get("output_path", 0))]
        return [{"node_id": c["node_id"], "p_idx": c["p_idx"], "depth": c["depth"], "pred": c["pred"],
                 "ctx": c["ctx"], "support_probs": c["support_probs"], "selected": c["node_id"] in path}
                for c in item["candidates"]]
    # normal output file: the chosen single-segment answer only (intermediate = [root placeholder, generation])
    inter, docs = item.get("intermediate") or [], item.get("docs") or []
    if len(inter) == 2 and len(docs) == 1:
        return [{"node_id": None, "p_idx": None, "depth": 1, "pred": inter[1], "ctx": docs[0],
                 "support_probs": None, "selected": True}]
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", required=True, help="output file (with or without --save_candidates)")
    parser.add_argument("--split", default="", help="comma-separated split files; default: every question in --f")
    parser.add_argument("--out", default=None, help="default: <f>.verdicts.jsonl")
    parser.add_argument("--alce_dir", default=os.environ.get("ALCE_DIR", os.path.expanduser("~/ALCE")))
    parser.add_argument("--max_questions", type=int, default=None, help="quick test on the first N questions")
    args = parser.parse_args()

    sys.path.insert(0, args.alce_dir)
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import eval as alce_eval
    from utils import get_max_memory

    # Same model, dtype and prompt as ALCE's compute_autoais
    alce_eval.autoais_model = AutoModelForSeq2SeqLM.from_pretrained(
        alce_eval.AUTOAIS_MODEL, torch_dtype=torch.bfloat16, max_memory=get_max_memory(), device_map="auto")
    alce_eval.autoais_tokenizer = AutoTokenizer.from_pretrained(alce_eval.AUTOAIS_MODEL, use_fast=False)

    outputs = load_outputs(args.f)
    q_ids = load_split_indices(args.split) if args.split else sorted(outputs)
    q_ids = [q for q in q_ids if q in outputs][:args.max_questions]
    out_path = args.out or args.f + ".verdicts.jsonl"

    n_rows, n_mismatch = 0, 0
    with open(out_path, "w") as f:
        for n, q in enumerate(q_ids):
            for cand in candidate_records(outputs[q]):
                claims = graded_claims(cand["pred"])
                probs = cand["support_probs"]
                if probs is not None and len(probs) != len(claims):
                    n_mismatch += 1  # text and token positions disagree; keep the pairs we can align
                passage = "Title: %s\n%s" % (cand["ctx"].get("title", ""), cand["ctx"].get("text", ""))
                for k, (claim, label) in enumerate(claims):
                    if not claim:
                        continue
                    row = {"q_idx": q, "node_id": cand["node_id"], "p_idx": cand["p_idx"], "depth": cand["depth"],
                           "selected": cand["selected"], "claim_idx": k, "claim": claim, "label": label,
                           "probs": probs[k] if probs is not None and k < len(probs) else None,
                           "verdict": alce_eval._run_nli_autoais(passage, claim)}
                    f.write(json.dumps(row) + "\n")
                    n_rows += 1
            if n % 20 == 0:
                print(f"[{n + 1}/{len(q_ids)}] {n_rows} claims checked", flush=True)
    print(f"done: {n_rows} claims from {len(q_ids)} questions, {n_mismatch} candidates with a label/probability "
          f"count mismatch")
    print("VERDICTS_FILE", out_path)


if __name__ == "__main__":
    main()
