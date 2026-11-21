"""Exploration N1.1b-i and N1.1c (CPU). Join the picked answer with up to K more of the existing answers, taken in
order of Division 2's support score and added only if that score is >= CUT. Reports str-em (exact), words, and
ESTIMATED citation recall/precision = sentence-weighted mean of the candidates' measured scores
(retrieval/predict_citations.py: within 0.4 points of full scoring on 12 files).
usage: python3 retrieval/gated_k.py"""
import json, re, string, random, statistics as st, ast
_P = set(string.punctuation)
def norm(s):
    s = s.lower(); s = "".join(ch for ch in s if ch not in _P); s = re.sub(r"\b(a|an|the)\b", " ", s); return " ".join(s.split())
def hits(text, qa):
    t = norm(text); return [any(norm(a) in t for a in q["short_answers"]) for q in qa]
TOK = r"\[(Retrieval|No Retrieval|Relevant|Irrelevant|Fully supported|Partially supported|No support / Contradictory|Utility:[1-5])\]"
def sentences(pred):
    p = re.sub(TOK, "", pred).replace("</s>", ""); return [s.strip() for s in p.split("[Continue to Use Evidence]") if s.strip()]
def ci(v, B=2000):
    r = random.Random(0); n = len(v); ms = sorted(st.mean(r.choices(v, k=n)) for _ in range(B))
    return "%+6.2f [%+6.2f, %+6.2f]" % (st.mean(v), ms[int(.025 * B)], ms[int(.975 * B)])
def lit(x): return ast.literal_eval(x) if isinstance(x, str) else x
RR = "results/runs/retrieval"; cands = {}; T = {}; GR = {}
for f in ("asqa_rrR2K20_0_end.json", "asqa_rrR2K20C_0_end.json"):
    for it in json.load(open(f"{RR}/{f}"))["data"]: cands[int(it["q_idx"])] = it
for s in ("AB", "C"):
    for l in open(f"{RR}/cand_metrics_rr_{s}_exact.jsonl"):
        r = json.loads(l); T[(int(r["q_idx"]), r["p_idx"])] = r
    for l in open(f"{RR}/cand_ground_rr_{s}.jsonl"):
        r = json.loads(l); GR.setdefault((int(r["q_idx"]), r["node_id"]), []).append(r["p_ent"])
C = set(json.load(open("critique/splits/asqa_C.json"))["indices"])
KEYS = ("str_em", "words", "cit_rec", "cit_prec", "passages")
def build(a, cut, K, order="ground"):
    q = int(a["q_idx"]); item = cands[q]; qa = item["qa_pairs"]; c = int(a["chosen_p_idx"])
    d1 = [x for x in item["candidates"] if x["depth"] == 1]; used = next(x for x in d1 if x["p_idx"] == c)
    seen = [norm(s) for s in sentences(used["pred"])]; plain = re.sub(r"\[\d+\]", "", a["output"])
    parts = [(T[(q, c)]["cit_sents"], T[(q, c)])]
    gs = lambda x: st.mean(GR.get((q, x["node_id"]), [0.0]))
    for x in sorted((x for x in d1 if x["p_idx"] != c), key=(lambda x: -gs(x)) if order == "ground" else (lambda x: x["p_idx"])):
        if len(parts) > K: break
        if gs(x) < cut: continue
        ss = [s for s in sentences(x["pred"]) if norm(s) and not any(norm(s) in y or y in norm(s) for y in seen)]
        if not ss: continue
        seen += [norm(s) for s in ss]; plain += " " + " ".join(ss); t = T[(q, x["p_idx"])]
        parts.append((min(len(ss), t["cit_sents"]) or 1, t))
    n = sum(w for w, _ in parts); h = hits(plain, qa)
    return dict(str_em=100 * sum(h) / max(len(qa), 1), words=len(plain.split()), passages=len(parts),
                cit_rec=sum(w * t["citation_rec"] for w, t in parts) / n, cit_prec=sum(w * t["citation_prec"] for w, t in parts) / n)
for row in ("c2", "d2v2"):
    picks = [a for a in json.load(open(f"{RR}/stack/RR_{row}_948.json"))["data"] if "chosen_p_idx" in a and (int(a["q_idx"]), int(a["chosen_p_idx"])) in T]
    base = {int(a["q_idx"]): build(a, 2.0, 0) for a in picks}
    print(f"\n######## first answer by {row.upper()}  (n={len(picks)}; Set C n={sum(q in C for q in base)})")
    print("  setting                  " + "  ".join(f"{k:>8s}" for k in KEYS) + "   | Set C, minus one passage: str-em, est. citation recall")
    for cut, K in ((2.0, 0), (0.25, 1), (0.5, 1), (0.75, 1), (0.9, 1), (0.5, 2), (0.75, 2), (0.9, 2), (0.5, 4), (0.75, 4), (0.9, 4)):
        R = {int(a["q_idx"]): build(a, cut, K) for a in picks}; qs = sorted(R); qc = [q for q in qs if q in C]
        lab = "one passage" if K == 0 else f"cut {cut:.2f}, up to {K} more"
        print(f"  {lab:24s} " + "  ".join(f"{st.mean(R[q][k] for q in qs):8.2f}" for k in KEYS) + ("" if K == 0 else
              f"   | {ci([R[q]['str_em'] - base[q]['str_em'] for q in qc])}  {ci([R[q]['cit_rec'] - base[q]['cit_rec'] for q in qc])}"))

    print("  N1.2a: same filter, candidates tried in the reranker's order instead of the checker's order")
    for cut, K in ((0.5, 1), (0.75, 1), (0.5, 2), (0.75, 2)):
        R = {int(a["q_idx"]): build(a, cut, K, "rank") for a in picks}; G = {int(a["q_idx"]): build(a, cut, K) for a in picks}; qs = sorted(R); qc = [q for q in qs if q in C]
        print(f"  {'cut %.2f, up to %d more' % (cut, K):24s} " + "  ".join(f"{st.mean(R[q][k] for q in qs):8.2f}" for k in KEYS) +
              f"   | {ci([R[q]['str_em'] - base[q]['str_em'] for q in qc])}  {ci([R[q]['cit_rec'] - base[q]['cit_rec'] for q in qc])}"
              f"   | minus checker order: str-em {ci([R[q]['str_em'] - G[q]['str_em'] for q in qc])}  cit {ci([R[q]['cit_rec'] - G[q]['cit_rec'] for q in qc])}")
