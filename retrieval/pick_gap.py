"""Exploration N1.2 (CPU). The Level 3 entry measurement found a 16-point gap between the passage the answer uses and
the best single passage of the five. Which available signal comes closest to finding the best passage?
For each way of choosing ONE passage: share of gold short answers in that passage (coverage), and str-em of the
answer written from it (both already measured per candidate). usage: python3 retrieval/pick_gap.py"""
import json, re, string, random, statistics as st, sys
sys.path.insert(0, "critique"); import learn_weights as LW
_P = set(string.punctuation)
def norm(s):
    s = s.lower(); s = "".join(ch for ch in s if ch not in _P); s = re.sub(r"\b(a|an|the)\b", " ", s); return " ".join(s.split())
def ci(v, B=2000):
    r = random.Random(0); n = len(v); ms = sorted(st.mean(r.choices(v, k=n)) for _ in range(B))
    return "%6.2f [%6.2f, %6.2f]" % (st.mean(v), ms[int(.025 * B)], ms[int(.975 * B)])
RR = "results/runs/retrieval"; cands = {}; T = {}; GR = {}; QR = {}
for f in ("asqa_rrR2K20_0_end.json", "asqa_rrR2K20C_0_end.json"):
    for it in json.load(open(f"{RR}/{f}"))["data"]: cands[int(it["q_idx"])] = it
for s in ("AB", "C"):
    for l in open(f"{RR}/cand_metrics_rr_{s}_exact.jsonl"):
        r = json.loads(l); T[(int(r["q_idx"]), r["p_idx"])] = r
    for l in open(f"{RR}/cand_ground_rr_{s}.jsonl"):
        r = json.loads(l); GR.setdefault((int(r["q_idx"]), r["node_id"]), []).append(r["p_ent"])
    for l in open(f"{RR}/cand_qarel_rr_{s}.jsonl"):
        r = json.loads(l); QR[(int(r["q_idx"]), r["node_id"])] = r["score"]
CFG = {k: json.load(open(f"results/critique_paper_scoring_configs/{f}.json")) for k, f in (("C1", "released"), ("C2", "paper_intended"))}
d2 = {int(a["q_idx"]): int(a["chosen_p_idx"]) for a in json.load(open(f"{RR}/stack/RR_d2v2_948.json"))["data"] if "chosen_p_idx" in a}
rng = random.Random(0); rows = {}
for q, item in cands.items():
    d1 = sorted((x for x in item["candidates"] if x["depth"] == 1), key=lambda x: x["p_idx"])
    if len(d1) != 5 or q not in d2 or any((q, x["p_idx"]) not in T for x in d1): continue
    qa = [[norm(a) for a in p["short_answers"]] for p in item["qa_pairs"]]
    cov = {x["p_idx"]: 100 * st.mean(any(a in norm(x["ctx"]["text"]) for a in sa) for sa in qa) for x in d1}
    em = {x["p_idx"]: T[(q, x["p_idx"])]["str_em"] for x in d1}
    def by(score): return max(d1, key=lambda x: (score(x), -x["p_idx"]))["p_idx"]
    def sr(k):
        W = CFG[k]["weights"]; wv = (W["w_rel"], W["w_sup"], W["w_use"], float(W["use_seqscore"]))
        return lambda x: sum(w * v for w, v in zip(wv, LW.features(x, CFG[k]["temperature"])))
    pick = {"reranker rank 1": 0, "random of the 5": rng.choice(d1)["p_idx"], "Self-RAG relevance token only": by(lambda x: x["relevance_score"]),
            "question-answer relevance (bge-base)": by(lambda x: QR.get((q, x["node_id"]), -9)), "support checker only": by(lambda x: st.mean(GR.get((q, x["node_id"]), [0]))),
            "C1 released score": by(sr("C1")), "C2 corrected score": by(sr("C2")), "D2-v2": d2[q],
            "longest answer": by(lambda x: T[(q, x["p_idx"])]["length"]),
            "ORACLE best passage by coverage": max(d1, key=lambda x: (cov[x["p_idx"]], em[x["p_idx"]]))["p_idx"], "ORACLE best answer by str-em": max(d1, key=lambda x: em[x["p_idx"]])["p_idx"]}
    rows[q] = {k: (cov[p], em[p], T[(q, p)]["length"], p) for k, p in pick.items()}
print("questions:", len(rows))
print(f"{'one passage chosen by':38s} {'coverage of that passage':24s} {'str-em of its answer':24s} words  mean rank")
for k in next(iter(rows.values())):
    print(f"{k:38s} {ci([r[k][0] for r in rows.values()]):24s} {ci([r[k][1] for r in rows.values()]):24s} {st.mean(r[k][2] for r in rows.values()):5.1f}  {1 + st.mean(r[k][3] for r in rows.values()):.2f}")
print("\nmean coverage and str-em by reranker position:")
pos = {p: ([], []) for p in range(5)}
for q, item in cands.items():
    if q not in rows: continue
    qa = [[norm(a) for a in pp["short_answers"]] for pp in item["qa_pairs"]]
    for x in item["candidates"]:
        if x["depth"] == 1:
            pos[x["p_idx"]][0].append(100 * st.mean(any(a in norm(x["ctx"]["text"]) for a in sa) for sa in qa)); pos[x["p_idx"]][1].append(T[(q, x["p_idx"])]["str_em"])
for p in range(5): print(f"  position {p + 1}: coverage {st.mean(pos[p][0]):5.2f}   answer str-em {st.mean(pos[p][1]):5.2f}   answer/coverage {st.mean(pos[p][1]) / st.mean(pos[p][0]):.2f}")

print("\nall metrics of the answer from each choice (lookup in the per-candidate table), and paired difference from reranker rank 1:")
M = ("str_em", "rougeLsum", "citation_rec", "citation_prec", "length")
cq = {q: {x["p_idx"]: T[(q, x["p_idx"])] for x in cands[q]["candidates"] if x["depth"] == 1} for q in rows}
for k in ("reranker rank 1", "C1 released score", "C2 corrected score", "D2-v2", "support checker only"):
    print(f"  {k:24s} " + "  ".join(f"{m} {st.mean(cq[q][rows[q][k][3]][m] for q in rows):6.2f}" for m in M))
C = set(json.load(open("critique/splits/asqa_C.json"))["indices"])
for g, qs in (("ALL", sorted(rows)), ("Set C", [q for q in sorted(rows) if q in C])):
    for k in ("C1 released score", "C2 corrected score", "D2-v2"):
        print(f"  {g:5s} rank 1 minus {k:20s} " + "  ".join(f"{m} {ci([cq[q][0][m] - cq[q][rows[q][k][3]][m] for q in qs])}" for m in M))
