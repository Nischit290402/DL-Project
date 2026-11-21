"""Score every (question, passage) pair of an ALCE-format file with a cross-encoder reranker.
Writes one JSON line per question: {"q_idx": i, "scores": [one float per passage, in the file's passage order]}.
The reranker sees the question and the passage only. Gold answers are never read.
usage: python rerank_score.py INPUT.json MODEL_ID OUT.jsonl"""
import json, sys, time, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

inp, model_id, out = sys.argv[1], sys.argv[2], sys.argv[3]
data = json.load(open(inp))
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id).cuda().half().eval()
print("model", model_id, "| commit", getattr(model.config, "_commit_hash", None), "| gpu", torch.cuda.get_device_name(0), flush=True)

t0 = time.time()
with open(out, "w") as f, torch.no_grad():
    for i, item in enumerate(data):
        passages = [d.get("title", "") + " " + d["text"] for d in item["docs"]]
        scores = []
        for s in range(0, len(passages), 50):
            chunk = passages[s:s + 50]
            enc = tok([item["question"]] * len(chunk), chunk, truncation=True, max_length=512,
                      padding=True, return_tensors="pt").to("cuda")
            scores += model(**enc).logits.float().view(-1).tolist()
        f.write(json.dumps({"q_idx": i, "scores": [round(x, 5) for x in scores]}) + "\n")
        if i % 100 == 0:
            print(f"[{i}/{len(data)}] {time.time() - t0:.0f}s", flush=True)
el = time.time() - t0
print(f"DONE {len(data)} questions in {el:.1f}s = {el / len(data):.3f} s/question (all passages in the file)")
