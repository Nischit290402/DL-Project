"""Check (exploration N1.1b-ii): can the citation scores of a joined answer be computed from the per-candidate
scores already measured? In a joined answer every sentence cites the passage it was written from, so its citation
recall should be the sentence-weighted mean of the candidates' own citation recall (same for precision).
Compares that prediction with the full ALCE scoring of the two-passage files. usage: python3 retrieval/predict_citations.py"""
import json, ast, statistics as st, glob
RR = "results/runs/retrieval"
T = {}
for f in ("cand_metrics_rr_AB_exact.jsonl", "cand_metrics_rr_C_exact.jsonl"):
    for l in open(f"{RR}/{f}"):
        r = json.loads(l); T[(int(r["q_idx"]), r["p_idx"])] = r
def lit(x): return ast.literal_eval(x) if isinstance(x, str) else x
for f in sorted(glob.glob(f"{RR}/stack/RR_*_two_*_948.json")):
    import os
    if not os.path.exists(f + ".perq.jsonl"): continue
    perq = {int(json.loads(l)["q_idx"]): json.loads(l) for l in open(f + ".perq.jsonl")}
    d = {"citation_rec": [], "citation_prec": []}; exact = {"citation_rec": 0, "citation_prec": 0}; n = 0
    for a in json.load(open(f))["data"]:
        q = int(a["q_idx"])
        if "second_p_idx" not in a or (q, int(a["chosen_p_idx"])) not in T: continue
        c1, c2 = T[(q, int(a["chosen_p_idx"]))], T[(q, int(a["second_p_idx"]))]
        n1 = c1["cit_sents"]; n2 = c2["cit_sents"] if a["output"].count("[2]") >= c2["cit_sents"] else a["output"].count("[2]"); n += 1
        for k in d:
            pred = (n1 * c1[k] + n2 * c2[k]) / (n1 + n2); d[k].append(pred - perq[q][k]); exact[k] += abs(pred - perq[q][k]) < 1e-6
    print(f"{f.split('/')[-1]:38s} two-passage answers {n:4d} | " + " | ".join(f"{k}: exact on {exact[k]:4d}, mean error {st.mean(v):+.3f}, mean abs error {st.mean(map(abs, v)):.3f}" for k, v in d.items()))
