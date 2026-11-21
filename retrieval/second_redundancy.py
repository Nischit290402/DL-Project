"""Exploration N1.1b-iii (CPU). The join drops a second-answer sentence only if it is contained in a first-answer
sentence word for word. How often does the filtered second answer still mostly repeat the first, and how much of
the gain remains if such repeats are dropped? Overlap = share of the second sentence's words (after normalising)
that already appear in the first answer. usage: python3 retrieval/second_redundancy.py"""
import json, re, string, random, statistics as st, sys
sys.path.insert(0, "critique"); import learn_weights as LW
_P = set(string.punctuation)
def norm(s):
    s = s.lower(); s = "".join(ch for ch in s if ch not in _P); s = re.sub(r"\b(a|an|the)\b", " ", s); return " ".join(s.split())
TOK = r"\[(Retrieval|No Retrieval|Relevant|Irrelevant|Fully supported|Partially supported|No support / Contradictory|Utility:[1-5])\]"
def sentences(pred):
    p = re.sub(TOK, "", pred).replace("</s>", ""); return [s.strip() for s in p.split("[Continue to Use Evidence]") if s.strip()]
def ci(v, B=2000):
    r = random.Random(0); n = len(v); ms = sorted(st.mean(r.choices(v, k=n)) for _ in range(B))
    return "%+6.2f [%+6.2f, %+6.2f]" % (st.mean(v), ms[int(.025 * B)], ms[int(.975 * B)])
RR = "results/runs/retrieval"; cands, T, GR = {}, {}, {}
for p in (f"{RR}/asqa_rrR2K20_0_end.json", f"{RR}/asqa_rrR2K20C_0_end.json"):
    for it in json.load(open(p))["data"]: cands[int(it["q_idx"])] = it
for s in ("AB", "C"):
    for l in open(f"{RR}/cand_metrics_rr_{s}_exact.jsonl"): r = json.loads(l); T[(int(r["q_idx"]), r["p_idx"])] = r
    for l in open(f"{RR}/cand_ground_rr_{s}.jsonl"): r = json.loads(l); GR.setdefault((int(r["q_idx"]), r["node_id"]), []).append(r["p_ent"])
CFG = json.load(open("results/critique_paper_scoring_configs/paper_intended.json")); C = set(json.load(open("critique/splits/asqa_C.json"))["indices"])
def em(plain, qa):
    t = norm(plain); return 100 * st.mean([any(norm(a) in t for a in p["short_answers"]) for p in qa]) if qa else 0.0
def run(maxov):
    R = {}; ovs = []
    for q, item in cands.items():
        d1 = [x for x in item["candidates"] if x["depth"] == 1]
        if len(d1) != 5 or any((q, x["p_idx"]) not in T for x in d1): continue
        qa = item["qa_pairs"]; used = LW.select([(c, LW.features(c, CFG["temperature"])) for c in LW.first_step(item)], CFG["weights"]); c = used["p_idx"]
        ss1 = sentences(used["pred"]); seen = [norm(s) for s in ss1]; w1 = set(" ".join(seen).split()); plain = " ".join(ss1); parts = [(T[(q, c)]["cit_sents"], T[(q, c)])]
        gs = lambda x: st.mean(GR.get((q, x["node_id"]), [0.0]))
        for x in sorted((x for x in d1 if x["p_idx"] != c), key=lambda x: -gs(x)):
            if gs(x) < 0.5: break
            ss = [s for s in sentences(x["pred"]) if norm(s) and not any(norm(s) in y or y in norm(s) for y in seen)]
            keep = []
            for s in ss:
                w = norm(s).split(); ov = sum(t in w1 for t in w) / max(len(w), 1); ovs.append(ov) if maxov > 1 else None
                if ov <= maxov: keep.append(s)
            if not keep: continue
            plain += " " + " ".join(keep); t = T[(q, x["p_idx"])]; parts.append((min(len(keep), t["cit_sents"]) or 1, t)); break
        n = sum(w for w, _ in parts) or 1
        R[q] = dict(one=em(" ".join(ss1), qa), str_em=em(plain, qa), words=len(plain.split()), two=len(parts) > 1,
                    cit_rec=sum(w * (m["citation_rec"] or 0) for w, m in parts) / n)
    return R, ovs
base, ovs = run(2.0); qs = sorted(base); qc = [q for q in qs if q in C]
ovs.sort(); print(f"second-answer sentences examined: {len(ovs)}; word overlap with the first answer: median {st.median(ovs):.2f}, share >= 0.6: {100 * st.mean(o >= .6 for o in ovs):.1f}%, share >= 0.8: {100 * st.mean(o >= .8 for o in ovs):.1f}%")
print("  max overlap allowed   str-em   words  est.cit.rec  with 2nd |  Set C: str-em gain over one passage     minus no-limit version")
for m in (2.0, 0.8, 0.6, 0.4):
    R, _ = run(m) if m < 2 else (base, None)
    print(f"  {('no limit' if m > 1 else '%.1f' % m):18s} {st.mean(R[q]['str_em'] for q in qs):8.2f} {st.mean(R[q]['words'] for q in qs):7.1f} {st.mean(R[q]['cit_rec'] for q in qs):9.2f} {100 * st.mean(R[q]['two'] for q in qs):8.0f}% | {ci([R[q]['str_em'] - R[q]['one'] for q in qc])}   {ci([R[q]['str_em'] - base[q]['str_em'] for q in qc]) if m < 2 else ''}")
