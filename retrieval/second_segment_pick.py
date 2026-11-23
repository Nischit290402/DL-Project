"""Choose a second segment and write ALCE-format answers (CPU).
Input: the first answers (PICKS, e.g. results/runs/combined/final_948.json), the continuations written by
self-rag/retrieval_lm/run_second_segment.py (SEG2 files) and their support scores.
A continuation is eligible if its evidence-sentence support is >= CUT (mean over its sentences of the best of
whole-passage and evidence-unit entailment). Among eligible continuations the one with the highest corrected
Self-RAG score (relevance + support + 0.5 utility + likelihood, paper weights) is appended, citing its passage as [2].
RULE: filter (the rule of the report) | beam (Self-RAG's own score only, no check) | none
(the whole* rules belong to an answer-level check that was tested and dropped; they need a WHOLE file and are not
used in the report)
Prints exact STR-EM and length for every rule plus an oracle; writes OUT for the chosen RULE.
usage: python3 retrieval/second_segment_pick.py PICKS OUT RULE CUT SEG2.json[,..] GROUND.jsonl[,..] GROUND_EV.jsonl[,..] [SPLIT.json|-] [WHOLE.jsonl,..] [CON] [ENT]"""
import json, re, string, sys, random, statistics as st
sys.path.insert(0, "critique"); sys.path.insert(0, "self-rag/retrieval_lm")
import learn_weights as LW
from rerank import build_output
_P = set(string.punctuation)
def norm(s):
    s = s.lower(); s = "".join(ch for ch in s if ch not in _P); s = re.sub(r"\b(a|an|the)\b", " ", s); return " ".join(s.split())
def em(text, qa):
    t = norm(text); return 100 * st.mean([any(norm(a) in t for a in q["short_answers"]) for q in qa]) if qa else 0.0
def ci(v, B=2000):
    r = random.Random(0); n = len(v); ms = sorted(st.mean(r.choices(v, k=n)) for _ in range(B))
    return "%+6.2f [%+6.2f, %+6.2f]" % (st.mean(v), ms[int(.025 * B)], ms[int(.975 * B)])
picks_f, out_f, RULE, CUT = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
seg = {}
for f in sys.argv[5].split(","):
    for it in json.load(open(f))["data"]: seg[int(it["q_idx"])] = it
GW, GE = {}, {}
for f in sys.argv[6].split(","):
    for l in open(f):
        r = json.loads(l)
        if r["sent_idx"] >= 0: GW.setdefault((int(r["q_idx"]), r["node_id"]), {})[r["sent_idx"]] = r["p_ent"]
for f in sys.argv[7].split(","):
    for l in open(f):
        r = json.loads(l); GE.setdefault((int(r["q_idx"]), r["node_id"]), {})[r["sent_idx"]] = max(r["t1"], r["t2"])
keep = set(json.load(open(sys.argv[8]))["indices"]) if len(sys.argv) > 8 and sys.argv[8] != "-" else None
WHOLE = {}                      # optional answer-level scores (dropped experiment): first segment vs each continuation
for f in (sys.argv[9].split(",") if len(sys.argv) > 9 else []):
    for l in open(f): r = json.loads(l); WHOLE[(int(r["q_idx"]), r["node_id"])] = r
CON = float(sys.argv[10]) if len(sys.argv) > 10 else 0.5; ENT = float(sys.argv[11]) if len(sys.argv) > 11 else 0.5
def support(q, node):
    w, e = GW.get((q, node), {}), GE.get((q, node), {}); ks = sorted(set(w) | set(e))
    return st.mean(max(w.get(k, 0.0), e.get(k, 0.0)) for k in ks) if ks else 0.0
out, R, STAT = [], {}, []
for a in json.load(open(picks_f))["data"]:
    q = int(a["q_idx"]); y = dict(a)
    if q not in seg or (keep is not None and q not in keep):
        if keep is None or q in keep: out.append(y)
        continue
    item = seg[q]; qa = item["qa_pairs"]; plain1 = re.sub(r"\[\d+\]", "", a["output"]); opts = []
    for c in item["candidates"]:
        text = build_output(item, c)
        if not re.sub(r"\[\d+\]", "", text).strip(): continue
        if norm(re.sub(r"\[\d+\]", "", text)) in norm(plain1): continue
        rel, sup, use, seq = LW.features(c, 1.0)
        opts.append(dict(c=c, text=text.replace("[1]", "[2]"), plain=re.sub(r"\[\d+\]", "", text), score=rel + sup + 0.5 * use + seq, g=support(q, c["node_id"])))
    res = {"none": None, "beam": max(opts, key=lambda o: o["score"]) if opts else None}
    elig = [o for o in opts if o["g"] >= CUT]; res["filter"] = max(elig, key=lambda o: o["score"]) if elig else None
    res["filter_g"] = max(elig, key=lambda o: o["g"]) if elig else None
    wh = lambda o: WHOLE.get((q, o["c"]["node_id"]), {"con": 0.0, "ent": 0.0})
    ok_w = [o for o in elig if wh(o)["con"] < CON and wh(o)["ent"] < ENT]; res["whole"] = max(ok_w, key=lambda o: o["score"]) if ok_w else None
    ok_c = [o for o in elig if wh(o)["con"] < CON]; res["whole_con"] = max(ok_c, key=lambda o: o["score"]) if ok_c else None
    res["whole_div"] = max(elig, key=lambda o: (wh(o)["con"], o["score"])) if elig else None        # most different from segment 1
    dis = [o for o in elig if wh(o)["con"] >= CON]; res["whole_pref"] = max(dis, key=lambda o: o["score"]) if dis else res["filter"]
    res["only_dis"] = max(dis, key=lambda o: o["score"]) if dis else None
    ok_e = [o for o in elig if wh(o)["ent"] < ENT]; res["whole_ent"] = max(ok_e, key=lambda o: o["score"]) if ok_e else None
    STAT.append((len(opts), len(elig), len(ok_c), len(ok_e), len(ok_w), bool(res["filter"] and wh(res["filter"])["con"] >= CON), bool(res["filter"] and wh(res["filter"])["ent"] >= ENT)))
    res["oracle"] = max(opts, key=lambda o: em(plain1 + " " + o["plain"], qa)) if opts else None
    one = em(plain1, qa)
    R[q] = {k: dict(em=em(plain1 + " " + o["plain"], qa) if o else one, w=len((plain1 + " " + (o["plain"] if o else "")).split()), has=o is not None) for k, o in res.items()}
    o = res[RULE]
    if o: y["output"] = a["output"] + " " + o["text"]; y["docs"] = list(a["docs"]) + [o["c"]["ctx"]]; y["second_p_idx"] = o["c"]["p_idx"]
    out.append(y)
json.dump({"data": out}, open(out_f, "w"))
qs = sorted(R); print(f"questions with continuations: {len(qs)} | wrote {len(out)} answers to {out_f} with rule {RULE}, cut {CUT}")
if STAT: print("  continuations per question: written %.2f, pass support %.2f, also not contradicting %.2f, also not restating %.2f, pass all %.2f | the support-filter pick contradicts segment 1 on %.1f%% of questions, restates it on %.1f%%" % tuple([st.mean(x[i] for x in STAT) for i in range(5)] + [100 * st.mean(x[5] for x in STAT), 100 * st.mean(x[6] for x in STAT)]))
for k in ("none", "beam", "filter", "whole_con", "whole_ent", "whole", "whole_div", "whole_pref", "only_dis", "oracle"):
    print(f"  {k:9s} STR-EM {st.mean(R[q][k]['em'] for q in qs):6.2f}  words {st.mean(R[q][k]['w'] for q in qs):5.1f}  second segment on {100 * st.mean(R[q][k]['has'] for q in qs):3.0f}%   vs first answer only: {ci([R[q][k]['em'] - R[q]['none']['em'] for q in qs])}")
