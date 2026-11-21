"""Control for the reranking test: 5 passages drawn at random from the GTR top-K (fixed seed), kept in GTR order.
Separates "the reranker orders well" from "any 5 passages from a deeper pool".
usage: python build_random.py INPUT.json K SEED OUT.json"""
import json, random, sys
inp, K, seed, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
data = json.load(open(inp))
for i, item in enumerate(data):
    rng = random.Random(seed * 1000003 + i)                 # one independent draw per question
    pick = sorted(rng.sample(range(min(K, len(item["docs"]))), 5))
    item["docs"] = [dict(item["docs"][j], gtr_rank=j + 1) for j in pick]
json.dump(data, open(out, "w"))
print("wrote", out, "| K =", K, "| seed =", seed)
