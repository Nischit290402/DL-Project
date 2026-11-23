"""Evidence-sentence grounding (GPU). Same candidates, same answer sentences and same NLI model as
grounding/cand_ground.py, but the premise is one evidence unit of the candidate's passage instead of the whole
passage, and each answer sentence keeps its best-supported unit (max over units), as in fact-verification
pipelines (FEVER: select evidence sentences, then verify) and SummaC-ZS (Laban et al. 2022).

Why: grounding/nli_probe.py showed that the sentence-level NLI model labels 34% of near-verbatim answer sentences
as not entailed when the premise is the whole ~100-word passage, and entails 93% of them when the premise is the
matching passage sentence (results/nli_grounding_labels_and_probe.md, section 3).

Units scored for every answer sentence (the passage title is NOT in the unit unless stated):
  s1   each passage sentence            s2   each window of two consecutive sentences
  t1   "<title>. " + each sentence      t2   "<title>. " + each two-sentence window
Output: one JSON line per (candidate, answer sentence) with the max p_ent over units for s1, s2, t1, t2.

    python grounding/cand_ground_evidence.py --f results/runs/retrieval/asqa_rrR2K20C_0_end.json \
        --split critique/splits/asqa_C.json --out results/runs/retrieval/cand_ground_ev_rr_C.jsonl
"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "critique")); sys.path.insert(0, HERE)
from common import load_split_indices  # noqa: E402
from cand_ground import MODEL, hypotheses  # noqa: E402


def units(ctx, sent_tokenize):
    sents = [s.strip() for s in sent_tokenize(ctx.get("text", "")) if s.strip()]
    title = ctx.get("title", "").strip()
    out = [("s1", s) for s in sents] + [("s2", sents[i] + " " + sents[i + 1]) for i in range(len(sents) - 1)]
    out += [("t" + k[1], f"{title}. {u}") for k, u in list(out)] if title else []
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f", required=True); ap.add_argument("--split", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL); ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_questions", type=int, default=None)
    a = ap.parse_args()
    import torch
    from nltk import sent_tokenize
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    keep = set(load_split_indices(a.split))
    data = [it for it in json.load(open(a.f))["data"] if it["q_idx"] in keep][:a.max_questions]
    rows, prem, hyp, owner, kind = [], [], [], [], []
    for it in data:
        for c in sorted([c for c in it["candidates"] if c["depth"] == 1], key=lambda c: c["p_idx"]):
            us = units(c["ctx"], sent_tokenize)
            for k, s in hypotheses(it, c, sent_tokenize):
                if k < 0: continue
                rows.append({"q_idx": it["q_idx"], "node_id": c["node_id"], "p_idx": c["p_idx"], "sent_idx": k,
                             "s1": 0.0, "s2": 0.0, "t1": 0.0, "t2": 0.0, "n_units": len(us)})
                for uk, u in us:
                    prem.append(u); hyp.append(s); owner.append(len(rows) - 1); kind.append(uk)
    print("questions:", len(data), "answer sentences:", len(rows), "NLI inputs:", len(prem), flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(a.model).to(device).eval()
    ent = {v.lower(): k for k, v in model.config.id2label.items()}["entailment"]
    order = sorted(range(len(prem)), key=lambda i: len(prem[i]) + len(hyp[i]))      # similar lengths per batch
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(order), a.batch_size):
            idx = order[s:s + a.batch_size]
            enc = tok([prem[i] for i in idx], [hyp[i] for i in idx], truncation=True, max_length=512, padding=True,
                      return_tensors="pt").to(device)
            pe = model(**enc).logits.float().softmax(-1)[:, ent].tolist()
            for i, p in zip(idx, pe):
                r = rows[owner[i]]
                if p > r[kind[i]]: r[kind[i]] = round(p, 5)
            if (s // a.batch_size) % 200 == 0: print(f"[{s}/{len(order)}] {time.time() - t0:.0f}s", flush=True)
    with open(a.out, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(f"done in {time.time() - t0:.0f}s", flush=True); print("CAND_GROUND_EV_OK", a.out, flush=True)


if __name__ == "__main__":
    main()
