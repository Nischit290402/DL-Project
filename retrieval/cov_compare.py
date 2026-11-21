"""Component metrics for the 5 passages a question is given, compared with a baseline file on the same questions.
rank1  = answer coverage of the first passage        mean5  = mean coverage of the 5 passages, one at a time
best5  = coverage of the best single passage of 5    union5 = coverage of the 5 passages together
Coverage = share of gold sub-answers (qa_pairs) with a short answer found as a normalised substring (ALCE definition).
usage: python cov_compare.py BASE.json name=FILE.json [name=FILE.json ...] --splits A.json B.json C.json"""
import json, re, string, sys, random, statistics as st
_P = set(string.punctuation)
def norm(s):
    s = s.lower(); s = "".join(ch for ch in s if ch not in _P)
    s = re.sub(r"\b(a|an|the)\b", " ", s); return " ".join(s.split())
def ci(v, B=2000):
    r = random.Random(0); n = len(v); ms = sorted(st.mean(r.choices(v, k=n)) for _ in range(B))
    return st.mean(v), ms[int(.025 * B)], ms[int(.975 * B)]
def metrics(path):
    rows = []
    for x in json.load(open(path)):
        qa = [[norm(a) for a in q["short_answers"]] for q in x["qa_pairs"]]
        docs = x["docs"][:5]
        H = [[any(a in t for a in sa) for sa in qa] for t in (norm(d.get("title", "") + " " + d["text"]) for d in docs)]
        if not qa:
            rows.append(dict(rank1=0, mean5=0, best5=0, union5=0, new=0)); continue
        cov = [100 * st.mean(h) for h in H]
        rows.append(dict(rank1=cov[0], mean5=st.mean(cov), best5=max(cov),
                         union5=100 * st.mean(any(h[j] for h in H) for j in range(len(qa))),
                         new=sum(d.get("gtr_rank", k + 1) > 5 for k, d in enumerate(docs))))
    return rows
args = sys.argv[1:]; sp = args.index("--splits"); files, splits = args[:sp], args[sp + 1:]
base = metrics(files[0]); groups = {}
for s in splits:
    j = json.load(open(s)); groups[j.get("name", s.split("_")[-1].split(".")[0])] = j["indices"]
groups["ALL"] = list(range(len(base)))
KEYS = ["mean5", "rank1", "best5", "union5"]
for g, idx in groups.items():
    print(f"\n== Set {g} (n={len(idx)})")
    print("  %-10s " % "GTR top-5" + "  ".join("%s %5.1f [%4.1f,%4.1f]" % ((k,) + ci([base[i][k] for i in idx])) for k in KEYS))
    for spec in files[1:]:
        name, path = spec.split("=", 1); R = metrics(path)
        print("  %-10s " % name + "  ".join("%s %5.1f" % (k, st.mean(R[i][k] for i in idx)) for k in KEYS)
              + "  | of its 5 passages, from GTR rank>5: %.2f" % st.mean(R[i]["new"] for i in idx))
        print("  %-10s " % "  vs GTR" + "  ".join("%s %+5.1f [%+5.1f,%+5.1f]" % ((k,) + ci([R[i][k] - base[i][k] for i in idx])) for k in KEYS))
