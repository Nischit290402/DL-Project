"""Level 3 step 2: paired differences of the fully scored two-passage files from the single-passage row and from
the random-second control. usage: python3 retrieval/two_passage_compare.py"""
import json, random, statistics as st, os
D = "results/runs/retrieval/stack"; KEYS = ("str_em", "rougeLsum", "citation_rec", "citation_prec", "length")
def load(f): return {int(json.loads(l)["q_idx"]): json.loads(l) for l in open(f)}
def ci(v, B=2000):
    r = random.Random(0); n = len(v); ms = sorted(st.mean(r.choices(v, k=n)) for _ in range(B))
    return "%+6.2f [%+6.2f, %+6.2f]" % (st.mean(v), ms[int(.025 * B)], ms[int(.975 * B)])
C = json.load(open("critique/splits/asqa_C.json"))["indices"]
for row in ("c2", "d2v2"):
    single = load(f"{D}/RR_{row}_948.json.perq.jsonl"); T = {}
    print(f"\n######## first answer by {row.upper()}")
    print("  single   ", json.load(open(f"{D}/RR_{row}_948.json.summary.json")))
    for m in ("next", "c2score", "next_fs", "rand"):
        f = f"{D}/RR_{row}_two_{m}_948.json"
        if not os.path.exists(f + ".summary.json"): print("  missing", m); continue
        T[m] = load(f + ".perq.jsonl"); print(f"  {m:8s} ", json.load(open(f + ".summary.json")))
    for g, qs in (("C", C), ("ALL", sorted(single))):
        print(f"  == Set {g} (n={len(qs)})")
        for m, t in T.items():
            print(f"   {m:8s} minus single: " + "  ".join(f"{k} {ci([t[q][k] - single[q][k] for q in qs])}" for k in KEYS))
        for m in ("next", "c2score", "next_fs"):
            if m in T and "rand" in T:
                print(f"   {m:8s} minus rand:   " + "  ".join(f"{k} {ci([T[m][q][k] - T['rand'][q][k] for q in qs])}" for k in KEYS))
