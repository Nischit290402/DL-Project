"""Level 3 entry measurement (no GPU). For one answer file: how much gold answer content is in the passage the
answer was built from, in the best single passage of the five, in all five together, and in the answer itself.
Matching as in retrieval/coverage.py (ALCE normalisation + substring). Paired bootstrap, 2,000 resamples, seed 0.
usage: python3 retrieval/used_vs_unused.py PASSAGES.json K ANSWERS.json ANSWERS.perq.jsonl [SPLIT.json ...]"""
import json, re, string, sys, random, statistics as st, ast
_P = set(string.punctuation)
def norm(s):
    s = s.lower(); s = "".join(ch for ch in s if ch not in _P)
    s = re.sub(r"\b(a|an|the)\b", " ", s); return " ".join(s.split())
def ci(v, B=2000, seed=0):
    r = random.Random(seed); n = len(v); ms = sorted(st.mean(r.choices(v, k=n)) for _ in range(B))
    return "%6.2f [%6.2f, %6.2f]" % (st.mean(v), ms[int(.025 * B)], ms[int(.975 * B)])
def lit(x): return ast.literal_eval(x) if isinstance(x, str) else x
pas = json.load(open(sys.argv[1])); K = int(sys.argv[2])
ans = json.load(open(sys.argv[3])); ans = ans["data"] if isinstance(ans, dict) else ans
perq = {int(json.loads(l)["q_idx"]): json.loads(l) for l in open(sys.argv[4])}
rows = {}; skipped = []; mismatch = 0
for a in ans:
    qi = int(a["q_idx"]); inter = lit(a["intermediate"])
    if sum(1 for s in inter if s != "[Retrieval]") != 1: skipped.append(qi); continue
    x = pas[qi]; assert x["question"] == a["question"]
    docs = x["docs"][:K]; c = int(a["chosen_p_idx"])
    adocs = lit(a["docs"])
    if len(adocs) == K and adocs[c]["text"] != docs[c]["text"]: mismatch += 1
    qa = [[norm(s) for s in q["short_answers"]] for q in x["qa_pairs"]]
    if not qa: skipped.append(qi); continue
    H = [[any(s in norm(d["text"]) for s in sa) for sa in qa] for d in docs]
    cov = lambda idx: 100 * st.mean(any(H[i][j] for i in idx) for j in range(len(qa)))
    used = cov([c]); others = [i for i in range(K) if i != c]
    rows[qi] = dict(used=used, best1=max(cov([i]) for i in range(K)), union=cov(range(K)),
                    used_plus_best=max(cov([c, i]) for i in others), used_plus_next=cov([c, others[0]]),
                    best_pair=max(cov([i, j]) for i in range(K) for j in range(i + 1, K)),
                    answer=perq[qi]["str_em"], c=c, n_qa=len(qa))
print("answer file:", sys.argv[3], "| questions used:", len(rows), "| left out:", skipped, "| chosen-passage text mismatches:", mismatch)
groups = {"ALL": sorted(rows)}
for s in sys.argv[5:]:
    j = json.load(open(s)); groups[j.get("name", s)] = [i for i in j["indices"] if i in rows]
for g, idx in groups.items():
    R = [rows[i] for i in idx]; col = lambda k: [r[k] for r in R]; d = lambda a, b: [r[a] - r[b] for r in R]
    print(f"\n== {g} (n={len(R)})")
    for k, lab in (("union", "all 5 passages together"), ("best_pair", "best two passages"), ("best1", "best single passage"),
                   ("used_plus_best", "used + best other passage"), ("used_plus_next", "used + next-ranked other"),
                   ("used", "passage used"), ("answer", "answer (str-em)")):
        print(f"  {lab:28s} {ci(col(k))}")
    for a, b in (("union", "used"), ("best1", "used"), ("used_plus_best", "used"), ("used_plus_next", "used"), ("used", "answer"), ("union", "best1")):
        print(f"  {a:>14s} - {b:<8s} {ci(d(a, b))}")
    print("  questions where the other 4 passages hold answer content the used one lacks: %.1f%%" % (100 * st.mean(r["union"] > r["used"] for r in R)))
    print("  questions where the used passage is the top-ranked one: %.1f%%" % (100 * st.mean(r["c"] == 0 for r in R)))
    print("  questions where the used passage already holds everything the 5 hold: %.1f%%" % (100 * st.mean(r["union"] == r["used"] for r in R)))
