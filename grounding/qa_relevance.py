"""Question-passage relevance of every depth-1 candidate's passage, from off-the-shelf rerankers (GPU, minutes).

Feasibility check for a relevance gate on D2: can a reranker tell when D2 moved to a passage that does not answer
the question (wrong entity / non-answer / other reading)? Scores (question, "title. text") for every candidate.
Output: one JSON line per (candidate, model): q_idx, node_id, p_idx, model, score (raw logit).

    python grounding/qa_relevance.py --f results/runs/asqa_cands_0_end.json \
        --split critique/splits/asqa_A.json,critique/splits/asqa_B.json --out results/runs/cand_qarel_AB.jsonl
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "critique"))
from common import load_split_indices  # noqa: E402

MODELS = ["cross-encoder/ms-marco-MiniLM-L-12-v2", "BAAI/bge-reranker-base"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=",".join(MODELS))
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    keep = set(load_split_indices(a.split))
    pairs = []
    for it in json.load(open(a.f))["data"]:
        if it["q_idx"] not in keep:
            continue
        for c in it["candidates"]:
            if c["depth"] == 1:
                pairs.append((it["q_idx"], c["node_id"], c["p_idx"], it["question"],
                              "%s. %s" % (c["ctx"].get("title", ""), c["ctx"].get("text", ""))))
    print("pairs", len(pairs), flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with open(a.out, "w") as f:
        for m in a.models.split(","):
            try:
                tok = AutoTokenizer.from_pretrained(m)
            except Exception as e:  # newer tokenizer.json than the pinned tokenizers can read
                print("fast tokenizer failed, using slow:", type(e).__name__, flush=True)
                tok = AutoTokenizer.from_pretrained(m, use_fast=False)
            model = AutoModelForSequenceClassification.from_pretrained(m).to(dev).eval()
            with torch.no_grad():
                for s in range(0, len(pairs), 32):
                    b = pairs[s:s + 32]
                    enc = tok([p[3] for p in b], [p[4] for p in b], truncation=True, max_length=512,
                              padding=True, return_tensors="pt").to(dev)
                    logits = model(**enc).logits.float()
                    sc = logits[:, 0] if logits.shape[-1] == 1 else logits.softmax(-1)[:, -1]
                    for p, v in zip(b, sc.tolist()):
                        f.write(json.dumps({"q_idx": p[0], "node_id": p[1], "p_idx": p[2], "model": m,
                                            "score": round(v, 5)}) + "\n")
            print("done", m, flush=True)
    print("QAREL_OK", a.out, flush=True)


if __name__ == "__main__":
    main()
