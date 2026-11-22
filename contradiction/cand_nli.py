"""Self-contradiction across candidates, GPU step: NLI over every pair of depth-1 candidates.

Self-RAG writes 5 candidate answers per ASQA question, one per retrieved passage, and ranks each in isolation.
This scores how strongly each candidate contradicts / entails each other candidate of the same question with
cross-encoder/nli-deberta-v3-large (not the T5-11B model ALCE uses to grade citations, to avoid ranking with
the grader). Both directions, two variants:
  noctx  premise = candidate i,                          hypothesis = candidate j
  ctx    premise = "Question: <q> Answer: <candidate i>", hypothesis = candidate j

Input: an output file made with --save_candidates (results/runs/asqa_cands_0_end.json, Set A + B).
Output: one JSON line per ordered pair and variant: q_idx, i, j (candidate p_idx), variant, p_con, p_ent, p_neu.

    python contradiction/cand_nli.py --f results/runs/asqa_cands_0_end.json \
        --split critique/splits/asqa_A.json --out results/runs/cand_nli_A.jsonl
Run through cluster/run_cand_nli.sbatch (GPU). Analysis: contradiction/diagnostics.py (laptop, stdlib only).
"""
import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "critique"))
from common import clean_claim, load_split_indices  # noqa: E402

MODEL = "cross-encoder/nli-deberta-v3-large"


def depth1(item):
    return sorted([c for c in item["candidates"] if c["depth"] == 1], key=lambda c: c["p_idx"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f", required=True)
    ap.add_argument("--split", default="critique/splits/asqa_A.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--batch_size", type=int, default=32)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    keep = set(load_split_indices(a.split))
    data = [it for it in json.load(open(a.f))["data"] if it["q_idx"] in keep]
    print("questions:", len(data), "of split", len(keep), flush=True)

    rows, prem, hyp = [], [], []
    for it in data:
        cands = depth1(it)
        texts = {c["p_idx"]: clean_claim(c["processed_pred"]) for c in cands}
        for ci, cj in itertools.permutations(cands, 2):
            for variant in ("noctx", "ctx"):
                p = texts[ci["p_idx"]]
                if variant == "ctx":
                    p = "Question: " + it["question"] + " Answer: " + p
                rows.append({"q_idx": it["q_idx"], "i": ci["p_idx"], "j": cj["p_idx"], "variant": variant})
                prem.append(p)
                hyp.append(texts[cj["p_idx"]])
    print("NLI inputs:", len(rows), flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # use_fast=False: the model's tokenizer.json needs tokenizers>=0.19; the env pins 0.15
    tok = AutoTokenizer.from_pretrained(a.model, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(a.model).to(device).eval()
    lab = {v.lower(): k for k, v in model.config.id2label.items()}
    print("labels:", model.config.id2label, flush=True)

    with torch.no_grad(), open(a.out, "w") as f:
        for s in range(0, len(rows), a.batch_size):
            enc = tok(prem[s:s + a.batch_size], hyp[s:s + a.batch_size], truncation=True, max_length=512,
                      padding=True, return_tensors="pt").to(device)
            probs = model(**enc).logits.float().softmax(-1).tolist()
            for r, p in zip(rows[s:s + a.batch_size], probs):
                r.update({"p_con": round(p[lab["contradiction"]], 5), "p_ent": round(p[lab["entailment"]], 5),
                          "p_neu": round(p[lab["neutral"]], 5)})
                f.write(json.dumps(r) + "\n")
    print("CAND_NLI_OK", a.out, flush=True)


if __name__ == "__main__":
    main()
