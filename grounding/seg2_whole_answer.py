"""Whole-answer check for two-segment answers (GPU, same NLI model as grounding/cand_ground.py).
The part-level checks judge a passage, a candidate, or a sentence against its own evidence. None of them reads the
assembled answer. Here the premise is the FIRST segment of the answer and the hypothesis is each sentence of a
candidate second segment:
  contradiction -> the second segment conflicts with what the answer already says
  entailment    -> the second segment restates the first (adds nothing)
  neutral       -> new content
Also scored in the other direction (second segment as premise, each first-segment sentence as hypothesis) for
contradiction only. Output: one JSON line per continuation: q_idx, node_id, p_idx, con (max contradiction in either
direction), con_fwd, con_bwd, ent (mean entailment of its sentences by the first segment), n_sents.

    python grounding/seg2_whole_answer.py --seg2 out/seg2_AB.json --picks results/runs/combined/final_948.json --out out/seg2_AB.whole.jsonl
"""
import argparse, json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "critique")); sys.path.insert(0, HERE)
from rerank import build_output  # noqa: E402
from cand_ground import MODEL  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg2", required=True); ap.add_argument("--picks", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL); ap.add_argument("--batch_size", type=int, default=64)
    a = ap.parse_args()
    import torch
    from nltk import sent_tokenize
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    strip = lambda t: " ".join(re.sub(r"\[\d+\]", "", t).split())
    first = {int(x["q_idx"]): strip(x["output"]) for x in json.load(open(a.picks))["data"]}
    rows, prem, hyp, owner, kind = [], [], [], [], []
    for it in json.load(open(a.seg2))["data"]:
        q = int(it["q_idx"]); f = first[q]; fs = [s for s in sent_tokenize(f) if s.strip()]
        for c in it["candidates"]:
            t = strip(build_output(it, c)); ss = [s for s in sent_tokenize(t) if s.strip()]
            rows.append({"q_idx": q, "node_id": c["node_id"], "p_idx": c["p_idx"], "n_sents": len(ss), "con_fwd": 0.0, "con_bwd": 0.0, "ent_sum": 0.0})
            for s in ss: prem.append(f); hyp.append(s); owner.append(len(rows) - 1); kind.append("fwd")
            for s in (fs if t else []): prem.append(t); hyp.append(s); owner.append(len(rows) - 1); kind.append("bwd")
    print("continuations:", len(rows), "NLI inputs:", len(prem), flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(a.model).to(device).eval()
    lab = {v.lower(): k for k, v in model.config.id2label.items()}
    with torch.no_grad():
        for s in range(0, len(prem), a.batch_size):
            enc = tok(prem[s:s + a.batch_size], hyp[s:s + a.batch_size], truncation=True, max_length=512, padding=True, return_tensors="pt").to(device)
            pr = model(**enc).logits.float().softmax(-1).tolist()
            for i, p in zip(range(s, s + len(pr)), pr):
                r = rows[owner[i]]; con, ent = p[lab["contradiction"]], p[lab["entailment"]]
                if kind[i] == "fwd": r["con_fwd"] = max(r["con_fwd"], con); r["ent_sum"] += ent
                else: r["con_bwd"] = max(r["con_bwd"], con)
    with open(a.out, "w") as f:
        for r in rows:
            r["con"] = round(max(r["con_fwd"], r["con_bwd"]), 5); r["ent"] = round(r.pop("ent_sum") / max(r["n_sents"], 1), 5)
            r["con_fwd"] = round(r["con_fwd"], 5); r["con_bwd"] = round(r["con_bwd"], 5); f.write(json.dumps(r) + "\n")
    print("SEG2_WHOLE_OK", a.out, flush=True)


if __name__ == "__main__":
    main()
