"""Component ablation of the combined system on Set C: all 8 on/off combinations of the three components.
  passages   GTR top 5 (off) / reranked top 5 (on)
  critique   released score (off) / the paper's full critique score (on)
  grounding  none (off) / D2 term alpha 0.75 with the D2-v2 relevance gate delta 1 (on)
Every cell is picked by combined/rerank_combined.py and scored by grounding/lookup_perq.py, unchanged.
Reports each cell, each component ALONE (vs C1), each component REMOVED from the final system, and each component's
effect averaged over the four settings of the other two. Paired bootstrap, 2,000 resamples, seed 0.
usage: python3 combined/factorial.py [OUT_DIR]"""
import json, os, random, statistics as st, subprocess, sys
O = sys.argv[1] if len(sys.argv) > 1 else "out/factorial"; os.makedirs(O, exist_ok=True)
R = "results/runs"; SPLIT = "critique/splits/asqa_C.json"; PY = sys.executable
P = {0: dict(cands=f"{R}/asqa_cands_C.json", metrics=f"{R}/cand_metrics_C_exact.jsonl", ground=f"{R}/cand_ground_C.jsonl", rel=f"{R}/cand_qarel_C.jsonl"),
     1: dict(cands=f"{R}/retrieval/asqa_rrR2K20C_0_end.json", metrics=f"{R}/combined/cand_metrics_rr_C_exact.jsonl",
             ground=f"{R}/combined/cand_ground_rr_C.jsonl", rel=f"{R}/combined/cand_qarel_rr_C.jsonl")}
KEYS = ("str_em", "rougeLsum", "citation_rec", "citation_prec", "length")
def ci(v, B=2000):
    r = random.Random(0); n = len(v); ms = sorted(st.mean(r.choices(v, k=n)) for _ in range(B))
    lo, hi = ms[int(.025 * B)], ms[int(.975 * B)]; return "%+6.2f [%+6.2f, %+6.2f]%s" % (st.mean(v), lo, hi, " *" if lo > 0 or hi < 0 else "  ")
cell = {}
for p in (0, 1):
    for c in (0, 1):
        for g in (0, 1):
            out = f"{O}/p{p}c{c}g{g}_C.json"; f = P[p]
            cmd = [PY, "combined/rerank_combined.py", "--f", f["cands"], "--split", SPLIT, "--critique", "paper" if c else "released", "--out", out]
            if g: cmd += ["--ground", f["ground"], "--alpha", "0.75", "--rel", f["rel"], "--delta", "1"]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            subprocess.run([PY, "grounding/lookup_perq.py", "--f", out, "--metrics", f["metrics"]], check=True, stdout=subprocess.DEVNULL)
            cell[(p, c, g)] = {int(json.loads(l)["q_idx"]): json.loads(l) for l in open(out + ".perq.jsonl")}
qs = sorted(set.intersection(*[set(v) for v in cell.values()]))
qs = [q for q in qs if all(cell[k][q].get(m) is not None for k in cell for m in KEYS)]
name = lambda k: "+".join(n for n, on in zip(("retrieval", "critique", "grounding"), k) if on) or "C1 (none)"
print(f"Set C, questions with all 8 cells scored: {len(qs)}  (multi-segment questions are not in the lookup tables)\n")
print(f"{'components on':34s} " + " ".join(f"{m:>13s}" for m in KEYS))
for k in sorted(cell, key=lambda k: (sum(k), [-x for x in k])):
    print(f"{name(k):34s} " + " ".join(f"{st.mean(cell[k][q][m] for q in qs):13.2f}" for m in KEYS))
def diff(a, b, m): return ci([cell[a][q][m] - cell[b][q][m] for q in qs])
def table(title, rows):
    print(f"\n{title}\n{'':24s} " + " ".join(f"{m:>25s}" for m in KEYS[:4]))
    for lab, f in rows: print(f"{lab:24s} " + " ".join(f"{f(m):>25s}" for m in KEYS[:4]))
comps = (("retrieval", 0), ("critique", 1), ("grounding", 2))
on = lambda i: tuple(1 if j == i else 0 for j in range(3)); off = lambda i: tuple(0 if j == i else 1 for j in range(3))
table("Each component ALONE, against C1:", [(n, (lambda m, i=i: diff(on(i), (0, 0, 0), m))) for n, i in comps])
table("Each component REMOVED from the final system (final minus final-without-it):", [(n, (lambda m, i=i: diff((1, 1, 1), off(i), m))) for n, i in comps])
def main_effect(i, m):
    others = [k for k in cell if k[i] == 0]
    return ci([st.mean(cell[tuple(1 if j == i else k[j] for j in range(3))][q][m] - cell[k][q][m] for k in others) for q in qs])
table("Each component's effect AVERAGED over the 4 settings of the other two:", [(n, (lambda m, i=i: main_effect(i, m))) for n, i in comps])
table("All three together, against C1:", [("final", lambda m: diff((1, 1, 1), (0, 0, 0), m))])
