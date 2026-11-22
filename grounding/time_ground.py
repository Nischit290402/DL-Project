"""Extra seconds per question for D2's DeBERTa grounding of 5 candidates (pre-registered "also reported").

Same inputs and model setting as grounding/cand_ground.py (fp32, slow tokenizer, max_length 512, whole own passage as
premise). Two timings, model load excluded, after a warm-up, with cuda.synchronize around each timed call:
  batched      : all Set C NLI inputs in batches of --batch_size (offline throughput, as cand_ground.py ran)
  per_question : one question at a time, its 5 candidates' sentences in one batch (online latency per question)
Writes no scores, only timings; also checks p_ent against the stored file.

    python grounding/time_ground.py --f results/runs/asqa_cands_C.json --split critique/splits/asqa_C.json \
        --ref results/runs/cand_ground_C.jsonl --out results/runs/time_ground_C.json
"""
import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "critique"))
sys.path.insert(0, HERE)
from common import load_split_indices  # noqa: E402
from cand_ground import MODEL, hypotheses, passage_text  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--ref", help="stored cand_ground jsonl to compare p_ent with")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    a = ap.parse_args()

    import torch
    from nltk import sent_tokenize
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    keep = set(load_split_indices(a.split))
    data = [it for it in json.load(open(a.f))["data"] if it["q_idx"] in keep]
    per_q = []  # list of (q_idx, [(premise, hypothesis, key)])
    for it in data:
        items = []
        for c in sorted([c for c in it["candidates"] if c["depth"] == 1], key=lambda c: c["p_idx"]):
            p = passage_text(c["ctx"])
            for k, s in hypotheses(it, c, sent_tokenize):
                items.append((p, s, (it["q_idx"], c["node_id"], k)))
        per_q.append((it["q_idx"], items))
    flat = [x for _, items in per_q for x in items]
    print("questions:", len(per_q), "NLI inputs:", len(flat), flush=True)

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(dev).eval()
    ent = {v.lower(): k for k, v in model.config.id2label.items()}["entailment"]

    def run(batch):
        enc = tok([b[0] for b in batch], [b[1] for b in batch], truncation=True, max_length=512, padding=True,
                  return_tensors="pt").to(dev)
        return model(**enc).logits.float().softmax(-1)[:, ent].tolist()

    with torch.no_grad():
        for _, items in per_q[:5]:  # warm-up
            run(items)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        p_batched = []
        for s in range(0, len(flat), a.batch_size):
            p_batched += run(flat[s:s + a.batch_size])
        torch.cuda.synchronize()
        t_batched = time.perf_counter() - t0

        lat = []
        for _, items in per_q:
            torch.cuda.synchronize()
            t = time.perf_counter()
            run(items)
            torch.cuda.synchronize()
            lat.append(time.perf_counter() - t)

    res = {"gpu": torch.cuda.get_device_name(0), "model": MODEL, "precision": "fp32", "questions": len(per_q),
           "nli_inputs": len(flat), "inputs_per_question": len(flat) / len(per_q),
           "batched": {"batch_size": a.batch_size, "total_s": t_batched, "s_per_question": t_batched / len(per_q)},
           "per_question": {"mean_s": statistics.mean(lat), "median_s": statistics.median(lat),
                            "p90_s": sorted(lat)[int(0.9 * len(lat))], "max_s": max(lat)}}
    if a.ref:
        ref = {(r["q_idx"], r["node_id"], r["sent_idx"]): r["p_ent"] for r in map(json.loads, open(a.ref))}
        diffs = [abs(ref[x[2]] - p) for x, p in zip(flat, p_batched)]
        res["max_abs_diff_vs_stored_p_ent"] = max(diffs)
        res["n_compared"] = len(diffs)
    print(json.dumps(res, indent=1), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("TIME_GROUND_OK", flush=True)


if __name__ == "__main__":
    main()
