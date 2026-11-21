"""D2 NLI grounding, GPU step: does each candidate's OWN passage entail what the candidate says?

For every depth-1 candidate (5 per ASQA question), premise = the candidate's passage ("Title: <t>\n<text>",
the same passage string ALCE/T5 sees), hypothesis = each sentence of the answer Self-RAG would output if that
candidate were chosen. Sentences are split exactly as ALCE does for citation recall (build_output -> nltk
sent_tokenize -> remove_citations), so sentence k here is sentence k in ALCE's citation scoring. One extra row per
candidate with sent_idx = -1 uses the whole answer as hypothesis (for a "full" aggregation variant).

Model: cross-encoder/nli-deberta-v3-large (same as contradiction/cand_nli.py), deliberately NOT ALCE's T5-11B grader.
Labels are read from model.config.id2label (this model's order is contradiction, entailment, neutral).

Output: one JSON line per (candidate, sentence): q_idx, node_id, p_idx, sent_idx, sentence, p_ent, p_neu, p_con.

    python grounding/cand_ground.py --f results/runs/asqa_cands_0_end.json \
        --split critique/splits/asqa_A.json --out results/runs/cand_ground_A.jsonl
    python grounding/cand_ground.py --f results/runs/asqa_cands_C.json \
        --split critique/splits/asqa_C.json --out results/runs/cand_ground_C.jsonl
Cluster: cluster/run_cand_ground.sbatch. Quick test: --max_questions 10.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "critique"))
sys.path.insert(0, HERE)
from common import load_split_indices  # noqa: E402
from rerank import build_output  # noqa: E402
from cand_metrics import remove_citations  # noqa: E402

MODEL = "cross-encoder/nli-deberta-v3-large"


def passage_text(ctx):
    return "Title: %s\n%s" % (ctx.get("title", ""), ctx.get("text", ""))


def hypotheses(item, cand, sent_tokenize):
    """ALCE's sentences of the answer this candidate would give, without citation marks, plus the whole answer."""
    out = build_output(item, cand).strip().split("\n")[0].replace("<|im_end|>", "")
    sents = [remove_citations(s).strip() for s in sent_tokenize(out)]
    sents = [(k, s) for k, s in enumerate(sents) if s]
    whole = " ".join(remove_citations(out).split())
    return sents + ([(-1, whole)] if whole else [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f", required=True, help="--save_candidates output file")
    ap.add_argument("--split", required=True, help="comma-separated split files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--fp16", action="store_true", help="half precision (default fp32)")
    ap.add_argument("--max_questions", type=int, default=None, help="quick test on the first N questions")
    a = ap.parse_args()

    import torch
    from nltk import sent_tokenize
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    keep = set(load_split_indices(a.split))
    data = [it for it in json.load(open(a.f))["data"] if it["q_idx"] in keep][:a.max_questions]
    print("questions:", len(data), "of split", len(keep), flush=True)

    rows, prem, hyp = [], [], []
    for it in data:
        for c in sorted([c for c in it["candidates"] if c["depth"] == 1], key=lambda c: c["p_idx"]):
            p = passage_text(c["ctx"])
            for k, s in hypotheses(it, c, sent_tokenize):
                rows.append({"q_idx": it["q_idx"], "node_id": c["node_id"], "p_idx": c["p_idx"], "sent_idx": k,
                             "sentence": s})
                prem.append(p)
                hyp.append(s)
    print("NLI inputs:", len(rows), flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # use_fast=False as in contradiction/cand_nli.py (the cluster env pins tokenizers 0.15)
    tok = AutoTokenizer.from_pretrained(a.model, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(a.model).to(device).eval()
    if a.fp16:
        model = model.half()
    lab = {v.lower(): k for k, v in model.config.id2label.items()}
    print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "",
          "labels:", model.config.id2label, flush=True)

    n_trunc, t0 = 0, time.time()
    with torch.no_grad(), open(a.out, "w") as f:
        for s in range(0, len(rows), a.batch_size):
            enc = tok(prem[s:s + a.batch_size], hyp[s:s + a.batch_size], truncation=True, max_length=512,
                      padding=True, return_tensors="pt").to(device)
            n_trunc += int((enc["attention_mask"].sum(-1) >= 512).sum())
            probs = model(**enc).logits.float().softmax(-1).tolist()
            for r, p in zip(rows[s:s + a.batch_size], probs):
                r.update({"p_ent": round(p[lab["entailment"]], 5), "p_neu": round(p[lab["neutral"]], 5),
                          "p_con": round(p[lab["contradiction"]], 5)})
                f.write(json.dumps(r) + "\n")
            if (s // a.batch_size) % 100 == 0:
                print(f"[{s + len(probs)}/{len(rows)}] {time.time() - t0:.0f}s", flush=True)
    secs = time.time() - t0
    print(f"done in {secs:.0f}s ({secs / max(len(data), 1):.3f} s/question for 5 candidates), "
          f"{n_trunc} inputs hit the 512-token limit", flush=True)
    print("CAND_GROUND_OK", a.out, flush=True)


if __name__ == "__main__":
    main()
