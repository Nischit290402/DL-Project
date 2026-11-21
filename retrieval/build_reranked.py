"""Build a Self-RAG input file whose passages are the GTR top-K reordered by reranker score.
Each passage keeps its original position as "gtr_rank" (1 = GTR's best). Ties keep GTR order.
usage: python build_reranked.py INPUT.json SCORES.jsonl K OUT.json"""
import json, sys
inp, scores_path, K, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
data = json.load(open(inp))
scores = {}
for line in open(scores_path):
    r = json.loads(line); scores[r["q_idx"]] = r["scores"]
assert len(scores) == len(data), (len(scores), len(data))
for i, item in enumerate(data):
    s = scores[i]; assert len(s) == len(item["docs"])
    pool = [dict(d, gtr_rank=j + 1, rerank_score=s[j]) for j, d in enumerate(item["docs"][:K])]
    pool.sort(key=lambda d: -d["rerank_score"])          # Python's sort is stable: ties stay in GTR order
    item["docs"] = pool[:20]
json.dump(data, open(out, "w"))
print("wrote", out, "| K =", K)
