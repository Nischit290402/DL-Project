"""Add a filtered second passage to ANY one-passage system on the reranked passages (no GPU, no new generation).
Input: an ALCE-format answer file whose items carry chosen_p_idx (e.g. results/runs/combined/final_948.json).
The first answer is kept exactly as it is. One more of the answers the run already wrote (from another passage) is
appended, citing [2], if Division 2's support score for it (mean DeBERTa entailment, grounding/cand_ground.py) is
>= CUT. Candidates are tried from the highest support score down; sentences already contained in the first answer
are dropped. CUT = 0.5 was fixed before any scoring (second_passage.py, mode ground_gate; this script reproduces
that file when given results/runs/retrieval/stack/RR_c2_948.json).
Prints str-em and words (exact) and the citation scores estimated from the per-candidate scores
(predict_citations.py); full scoring: cluster/run_critique_score.sbatch on OUT.
With EV files the support of a sentence is the larger of its whole-passage and evidence-sentence entailment.
usage: python3 retrieval/second_on_picks.py PICKS.json OUT.json [CUT] [EV_AB.jsonl,EV_C.jsonl]"""
import json, re, string, sys, random, statistics as st, ast
_P = set(string.punctuation)
def norm(s):
    s = s.lower(); s = "".join(ch for ch in s if ch not in _P); s = re.sub(r"\b(a|an|the)\b", " ", s); return " ".join(s.split())
def em(text, qa):
    t = norm(text); return 100 * st.mean([any(norm(a) in t for a in q["short_answers"]) for q in qa]) if qa else 0.0
TOK = r"\[(Retrieval|No Retrieval|Relevant|Irrelevant|Fully supported|Partially supported|No support / Contradictory|Utility:[1-5])\]"
def sentences(pred):
    p = re.sub(TOK, "", pred).replace("</s>", ""); return [s.strip() for s in p.split("[Continue to Use Evidence]") if s.strip()]
def ci(v, B=2000):
    r = random.Random(0); n = len(v); ms = sorted(st.mean(r.choices(v, k=n)) for _ in range(B))
    return "%+6.2f [%+6.2f, %+6.2f]" % (st.mean(v), ms[int(.025 * B)], ms[int(.975 * B)])
def lit(x): return ast.literal_eval(x) if isinstance(x, str) else x
picks_f, out_f = sys.argv[1], sys.argv[2]; CUT = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
EVF = sys.argv[4].split(",") if len(sys.argv) > 4 else []   # evidence-sentence scores (grounding/cand_ground_evidence.py)
RR = "results/runs/retrieval"; cands, T, GR = {}, {}, {}
for f in ("asqa_rrR2K20_0_end.json", "asqa_rrR2K20C_0_end.json"):
    for it in json.load(open(f"{RR}/{f}"))["data"]: cands[int(it["q_idx"])] = it
for s in ("AB", "C"):
    for l in open(f"{RR}/cand_metrics_rr_{s}_exact.jsonl"): r = json.loads(l); T[(int(r["q_idx"]), r["p_idx"])] = r
    for l in open(f"{RR}/cand_ground_rr_{s}.jsonl"): r = json.loads(l); GR.setdefault((int(r["q_idx"]), r["node_id"]), []).append(r["p_ent"])
EV = {}
for f in EVF:
    for l in open(f): r = json.loads(l); EV.setdefault((int(r["q_idx"]), r["node_id"]), {})[r["sent_idx"]] = max(r["t1"], r["t2"])
C = set(json.load(open("critique/splits/asqa_C.json"))["indices"]); out = []; R = {}
for a in json.load(open(picks_f))["data"]:
    q = int(a["q_idx"]); item = cands[q]; qa = item["qa_pairs"]; y = dict(a)
    multi = "chosen_p_idx" not in a or sum(1 for s in lit(a["intermediate"]) if s != "[Retrieval]") != 1 or (q, int(a["chosen_p_idx"])) not in T
    plain = re.sub(r"\[\d+\]", "", a["output"]); one = em(plain, qa)
    if multi: out.append(y); continue
    c = int(a["chosen_p_idx"]); d1 = [x for x in item["candidates"] if x["depth"] == 1]; used = next(x for x in d1 if x["p_idx"] == c)
    seen = [norm(s) for s in sentences(used["pred"])]; parts = [(T[(q, c)]["cit_sents"], T[(q, c)])]
    def gs(x):
        w = GR.get((q, x["node_id"]), []); e = EV.get((q, x["node_id"]), {})
        if not e: return st.mean(w) if w else 0.0
        return st.mean(max(w[k] if k < len(w) else 0.0, e[k]) for k in sorted(e))
    for x in sorted((x for x in d1 if x["p_idx"] != c), key=lambda x: -gs(x)):
        if gs(x) < CUT: break
        ss = [s for s in sentences(x["pred"]) if norm(s) and not any(norm(s) in t or t in norm(s) for t in seen)]
        if not ss: continue
        y["output"] = a["output"] + " " + " ".join((s[:-1] if s[-1] in ".!?" else s) + " [2]." for s in ss)
        y["docs"] = [used["ctx"], x["ctx"]]; y["second_p_idx"] = x["p_idx"]; plain += " " + " ".join(ss)
        t = T[(q, x["p_idx"])]; parts.append((min(len(ss), t["cit_sents"]) or 1, t)); break
    n = sum(w for w, _ in parts) or 1; out.append(y)
    R[q] = dict(one=one, two=em(plain, qa), w1=len(re.sub(r"\[\d+\]", "", a["output"]).split()), w2=len(plain.split()), has=len(parts) > 1,
                c1=T[(q, c)]["citation_rec"] or 0, c2=sum(w * (m["citation_rec"] or 0) for w, m in parts) / n,
                p1=T[(q, c)]["citation_prec"] or 0, p2=sum(w * (m["citation_prec"] or 0) for w, m in parts) / n)
json.dump({"data": out}, open(out_f, "w"))
qs = sorted(R); qc = [q for q in qs if q in C]; m = lambda k, S=qs: st.mean(R[q][k] for q in S)
print(f"picks: {picks_f} | cut-off {CUT} | questions {len(qs)} | a second passage added on {100 * m('has'):.0f}%")
print(f"  one passage : str-em {m('one'):.2f}  words {m('w1'):.1f}  citation recall {m('c1'):.2f}  precision {m('p1'):.2f}")
print(f"  with second : str-em {m('two'):.2f}  words {m('w2'):.1f}  citation recall {m('c2'):.2f}  precision {m('p2'):.2f}   (citations estimated)")
print(f"  Set C (n={len(qc)}) paired: str-em {ci([R[q]['two'] - R[q]['one'] for q in qc])} | est. citation recall {ci([R[q]['c2'] - R[q]['c1'] for q in qc])} | est. precision {ci([R[q]['p2'] - R[q]['p1'] for q in qc])} | words {ci([R[q]['w2'] - R[q]['w1'] for q in qc])}")
