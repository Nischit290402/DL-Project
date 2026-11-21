"""Level 3 smallest test (no GPU, no new generation). Each run already wrote one answer per passage. Take the answer a
picking rule chose and append one more of those answers, built from a different passage, citing it as [2].
How the second answer is chosen:
  next     the highest passage in the reranker's order that is not the one used
  c2score  the highest remaining candidate under the corrected Self-RAG score (critique configs, paper weights)
  rand     a random remaining candidate (seed 0): control for "any extra text helps"
  next_fs, c2score_fs  (added after the first full scoring showed lower citation scores) as next / c2score, but a second
           answer is added only if the model itself marked every sentence of it [Fully supported]; otherwise the
           answer stays single. No threshold, nothing tuned.
  ground, ground_gate  (exploration N1.1b) second answer = remaining candidate with the highest mean support probability
           from Division 2's checker (results of grounding/cand_ground.py); ground_gate adds it only if that mean is
           at least 0.5 (more likely supported than not; not tuned).
  oracle   the remaining candidate that adds most gold content: upper bound, not a system
Sentences of the second answer that repeat the first are dropped; if nothing new is left the next candidate in the
same order is tried. Rule fixed before running: choose between next and c2score by str-em on Set A+B (tie: next),
then read Set C. usage: python3 retrieval/second_passage.py OUT_DIR"""
import json, re, string, sys, random, statistics as st, ast
sys.path.insert(0, "critique"); import learn_weights as LW
_P = set(string.punctuation)
def norm(s):
    s = s.lower(); s = "".join(ch for ch in s if ch not in _P); s = re.sub(r"\b(a|an|the)\b", " ", s); return " ".join(s.split())
def hits(text, qa):
    t = norm(text); return [any(norm(a) in t for a in q["short_answers"]) for q in qa]
TOK = r"\[(Retrieval|No Retrieval|Relevant|Irrelevant|Fully supported|Partially supported|No support / Contradictory|Utility:[1-5])\]"
def sentences(pred):
    p = re.sub(TOK, "", pred).replace("</s>", "")
    return [s.strip() for s in p.split("[Continue to Use Evidence]") if s.strip()]
def ci(v, B=2000):
    r = random.Random(0); n = len(v); ms = sorted(st.mean(r.choices(v, k=n)) for _ in range(B))
    return "%+6.2f [%+6.2f, %+6.2f]" % (st.mean(v), ms[int(.025 * B)], ms[int(.975 * B)])
def lit(x): return ast.literal_eval(x) if isinstance(x, str) else x
RR = "results/runs/retrieval"; out_dir = sys.argv[1]
cands = {}
for f in ("asqa_rrR2K20_0_end.json", "asqa_rrR2K20C_0_end.json"):
    for it in json.load(open(f"{RR}/{f}"))["data"]: cands[int(it["q_idx"])] = it
CFG = json.load(open("results/critique_paper_scoring_configs/paper_intended.json")); W = CFG["weights"]
wv = (W["w_rel"], W["w_sup"], W["w_use"], float(W["use_seqscore"]))
A = set(json.load(open("critique/splits/asqa_A.json"))["indices"]) | set(json.load(open("critique/splits/asqa_B.json"))["indices"])
C = set(json.load(open("critique/splits/asqa_C.json"))["indices"])
GR = {}                                      # Division 2's DeBERTa support probability per candidate sentence
for f in ("cand_ground_rr_AB.jsonl", "cand_ground_rr_C.jsonl"):
    for l in open(f"{RR}/{f}"):
        r = json.loads(l); GR.setdefault((int(r["q_idx"]), r["node_id"]), []).append(r["p_ent"])
rng = random.Random(0)
for row in ("c2", "d2v2"):
    picks = json.load(open(f"{RR}/stack/RR_{row}_948.json"))["data"]
    res = {m: {} for m in ("single", "next", "c2score", "rand", "oracle", "next_fs", "c2score_fs", "ground", "ground_gate", "ground_gate25", "ground_gate75")}; files = {m: [] for m in ("next", "c2score", "rand", "next_fs", "c2score_fs", "ground", "ground_gate", "ground_gate25", "ground_gate75")}
    for a in picks:
        q = int(a["q_idx"]); item = cands[q]; qa = item["qa_pairs"]; first = a["output"]
        plain1 = re.sub(r"\[\d+\]", "", first); h1 = hits(plain1, qa); n = max(len(qa), 1)
        res["single"][q] = (100 * sum(h1) / n, len(plain1.split()), 1)
        multi = sum(1 for s in lit(a["intermediate"]) if s != "[Retrieval]") != 1 or "chosen_p_idx" not in a
        if multi:
            for m in res:
                res[m][q] = res["single"][q]
                if m in files: files[m].append(dict(a))
            continue
        d1 = [c for c in item["candidates"] if c["depth"] == 1]; c = int(a["chosen_p_idx"])
        used = next(x for x in d1 if x["p_idx"] == c); others = [x for x in d1 if x["p_idx"] != c]
        seen = [norm(s) for s in sentences(used["pred"])]
        def add(x):
            return [s for s in sentences(x["pred"]) if norm(s) and not any(norm(s) in y or y in norm(s) for y in seen)]
        sc = {x["p_idx"]: sum(w * v for w, v in zip(wv, LW.features(x, CFG["temperature"]))) for x in others}
        ro = others[:]; rng.shuffle(ro)
        orders = {"next": sorted(others, key=lambda x: x["p_idx"]), "c2score": sorted(others, key=lambda x: -sc[x["p_idx"]]), "rand": ro,
                  "oracle": sorted(others, key=lambda x: -sum(p or b for p, b in zip(h1, hits(" ".join(add(x)), qa))))}
        fs = lambda x: "[Fully supported]" in x["pred"] and "[Partially supported]" not in x["pred"] and "[No support" not in x["pred"]
        orders["next_fs"] = [x for x in orders["next"] if fs(x)]; orders["c2score_fs"] = [x for x in orders["c2score"] if fs(x)]
        gs = lambda x: st.mean(GR.get((q, x["node_id"]), [0.0]))
        orders["ground"] = sorted(others, key=lambda x: -gs(x)); orders["ground_gate"] = [x for x in orders["ground"] if gs(x) >= 0.5]
        orders["ground_gate25"] = [x for x in orders["ground"] if gs(x) >= 0.25]; orders["ground_gate75"] = [x for x in orders["ground"] if gs(x) >= 0.75]   # N1.1b-i: is 0.5 fragile?
        for m, order in orders.items():
            second = None
            if not multi:
                for x in order:
                    ss = add(x)
                    if ss: second = (x, ss); break
            if second is None:
                res[m][q] = res["single"][q]
                if m in files: files[m].append(dict(a))
                continue
            x, ss = second
            extra = " ".join((s[:-1] if s[-1] in ".!?" else s) + " [2]." for s in ss)
            plain = plain1 + " " + " ".join(ss); h = hits(plain, qa)
            res[m][q] = (100 * sum(h) / n, len(plain.split()), 2)
            if m in files:
                y = dict(a); y["output"] = first + " " + extra; y["docs"] = [used["ctx"], x["ctx"]]; y["second_p_idx"] = x["p_idx"]
                y["first_docs_check"] = None; files[m].append(y)
    print(f"\n######## first answer picked by {row.upper()} on reranked passages")
    for g, S in (("A+B", A), ("C", C), ("ALL", A | C)):
        qs = sorted(S & set(res["single"])); print(f"== Set {g} (n={len(qs)})")
        for m in res:
            em = [res[m][q][0] for q in qs]; d = [res[m][q][0] - res["single"][q][0] for q in qs]
            print(f"  {m:8s} str-em {st.mean(em):5.2f}  vs single {ci(d) if m != 'single' else '':28s} words {st.mean(res[m][q][1] for q in qs):5.1f}  passages {st.mean(res[m][q][2] for q in qs):.2f}  str-em per 100 words {100 * st.mean(em) / st.mean(res[m][q][1] for q in qs):5.1f}")
        for m in ("next", "c2score"):
            print(f"  {m:8s} minus rand: {ci([res[m][q][0] - res['rand'][q][0] for q in qs])}")
    for m, items in files.items():
        for y in items: y.pop("first_docs_check", None)
        json.dump({"data": items}, open(f"{out_dir}/RR_{row}_two_{m}_948.json", "w"))
    ex = files["next"][1]; print("example:", ex["output"][:500], "| docs:", len(lit(ex["docs"])))
p = json.load(open(f"{RR}/stack/RR_c2_948.json"))["data"][1]; print("docs in a single-passage file:", len(lit(p["docs"])), "| type:", type(p["docs"]).__name__, "| qa type:", type(p["qa_pairs"]).__name__)
r = json.load(open("data/reranked/asqa_R2_K20.json")); bad = sum(c["ctx"]["text"] != r[q]["docs"][c["p_idx"]]["text"] for q, it in cands.items() for c in it["candidates"] if c["depth"] == 1)
print("candidates whose passage is not the reranked passage at that position:", bad)
