"""Exploration N1.3 (CPU). Ablation of the optional configuration: passages (GTR original / RR reranked) x answer
(one passage picked by C2 / plus one more existing answer if Division 2's support score >= 0.5).
str-em and words are exact; citation scores are the sentence-weighted estimate checked in predict_citations.py.
usage: python3 retrieval/ablation_2x2.py [OUT_DIR]   (OUT_DIR: also write the four answer files for full scoring)"""
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
R, RR = "results/runs", "results/runs/retrieval"
F = {"GTR": dict(cands=[f"{R}/asqa_cands_0_end.json", f"{R}/asqa_cands_C.json"], metrics=[f"{R}/cand_metrics_AB_exact.jsonl", f"{R}/cand_metrics_C_exact.jsonl"],
                 ground=[f"{R}/cand_ground_A.jsonl", f"{R}/cand_ground_B.jsonl", f"{R}/cand_ground_C.jsonl"]),
     "RR": dict(cands=[f"{RR}/asqa_rrR2K20_0_end.json", f"{RR}/asqa_rrR2K20C_0_end.json"], metrics=[f"{RR}/cand_metrics_rr_AB_exact.jsonl", f"{RR}/cand_metrics_rr_C_exact.jsonl"],
                ground=[f"{RR}/cand_ground_rr_AB.jsonl", f"{RR}/cand_ground_rr_C.jsonl"])}
CFG = json.load(open("results/critique_paper_scoring_configs/paper_intended.json"))
C = set(json.load(open("critique/splits/asqa_C.json"))["indices"]); KEYS = ("str_em", "words", "cit_rec", "cit_prec")
G = {}; OUT = {}
for base, f in F.items():
    cands, T, GR = {}, {}, {}
    for p in f["cands"]:
        for it in json.load(open(p))["data"]: cands.setdefault(int(it["q_idx"]), it)
    for p in f["metrics"]:
        for l in open(p): r = json.loads(l); T[(int(r["q_idx"]), r["p_idx"])] = r
    for p in f["ground"]:
        for l in open(p): r = json.loads(l); GR.setdefault((int(r["q_idx"]), r["node_id"]), []).append(r["p_ent"])
    for q, item in cands.items():
        d1 = [x for x in item["candidates"] if x["depth"] == 1]
        if len(d1) != 5 or any((q, x["p_idx"]) not in T for x in d1): continue
        qa = item["qa_pairs"]; used = LW.select([(c, LW.features(c, CFG["temperature"])) for c in LW.first_step(item)], CFG["weights"]); c = used["p_idx"]
        gs = lambda x: st.mean(GR.get((q, x["node_id"]), [0.0]))
        for mode in ("one", "two"):
            ss1 = sentences(used["pred"]); seen = [norm(s) for s in ss1]; plain = " ".join(ss1); parts = [(T[(q, c)]["cit_sents"], T[(q, c)])]
            if mode == "two":
                for x in sorted((x for x in d1 if x["p_idx"] != c), key=lambda x: -gs(x)):
                    if gs(x) < 0.5: break
                    ss = [s for s in sentences(x["pred"]) if norm(s) and not any(norm(s) in y or y in norm(s) for y in seen)]
                    if not ss: continue
                    plain += " " + " ".join(ss); t = T[(q, x["p_idx"])]; parts.append((min(len(ss), t["cit_sents"]) or 1, t)); break
            if len(sys.argv) > 1:            # also write ALCE-format files for full scoring
                cite = lambda ss, k: " ".join((s[:-1] if s[-1] in ".!?" else s) + f" [{k}]." for s in ss)
                out = cite(ss1, 1) + (" " + cite(ss, 2) if len(parts) > 1 else "")
                OUT.setdefault((base, mode), []).append({"question": item["question"], "qa_pairs": qa, "annotations": item.get("annotations"), "answer": item.get("answer"),
                    "sample_id": item.get("sample_id"), "q_idx": q, "output": out, "docs": [used["ctx"]] + ([x["ctx"]] if len(parts) > 1 else []), "chosen_p_idx": c})
            n = sum(w for w, _ in parts) or 1; t = norm(plain)
            G.setdefault((base, mode), {})[q] = dict(str_em=100 * st.mean([any(norm(a) in t for a in p["short_answers"]) for p in qa]) if qa else 0.0,
                words=len(plain.split()), cit_rec=sum(w * (m["citation_rec"] or 0) for w, m in parts) / n, cit_prec=sum(w * (m["citation_prec"] or 0) for w, m in parts) / n, two=len(parts) > 1)
qs = sorted(set.intersection(*[set(v) for v in G.values()])); qc = [q for q in qs if q in C]
print(f"questions: {len(qs)} (Set C: {len(qc)})")
for k, v in G.items():
    print(f"  {k[0]:3s} {k[1]:3s} " + "  ".join(f"{m} {st.mean(v[q][m] for q in qs):6.2f}" for m in KEYS) + f"  answers with a second passage {100 * st.mean(v[q]['two'] for q in qs):.0f}%")
d = lambda a, b, m, S: ci([G[a][q][m] - G[b][q][m] for q in S])
print("\nSet C, paired differences (str-em | est. citation recall):")
for lab, a, b in (("reranking, one-passage answers", ("RR", "one"), ("GTR", "one")), ("reranking, two-passage answers", ("RR", "two"), ("GTR", "two")),
                  ("second passage, original passages", ("GTR", "two"), ("GTR", "one")), ("second passage, reranked passages", ("RR", "two"), ("RR", "one")),
                  ("both together", ("RR", "two"), ("GTR", "one"))):
    print(f"  {lab:36s} {d(a, b, 'str_em', qc)} | {d(a, b, 'cit_rec', qc)}")
inter = [(G[("RR", "two")][q]["str_em"] - G[("RR", "one")][q]["str_em"]) - (G[("GTR", "two")][q]["str_em"] - G[("GTR", "one")][q]["str_em"]) for q in qc]
print(f"  interaction (second-passage gain on RR minus on GTR), str-em: {ci(inter)}")

if len(sys.argv) > 1:
    for (base, mode), items in OUT.items():
        json.dump({"data": sorted(items, key=lambda y: y["q_idx"])}, open(f"{sys.argv[1]}/{base}_c2_{mode}_gate50_948.json", "w"))
