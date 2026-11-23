"""Report Table 6: a second segment added to the combined system, Set C. CPU, standard library only.
Every row is compared with the combined system on the same questions (paired bootstrap, 2,000 resamples, seed 0).
The per-question ALCE scores of each variant were written by cluster/run_critique_score.sbatch.
usage: python3 scripts/second_segment_table.py"""
import json, random, statistics as st
R = "results/runs"; D = f"{R}/combined/second"
K = ("str_em", "citation_rec", "citation_prec", "rougeLsum", "length")
def ci(v, B=2000):
    r = random.Random(0); n = len(v); ms = sorted(st.mean(r.choices(v, k=n)) for _ in range(B))
    return "%+6.2f [%+6.2f, %+6.2f]" % (st.mean(v), ms[int(.025 * B)], ms[int(.975 * B)])
perq = lambda name: {int(json.loads(l)["q_idx"]): json.loads(l) for l in open(f"{D}/{name}.json.perq.jsonl")}
S = json.load(open("critique/splits/asqa_C.json"))["indices"]
# combined system (one segment): per-candidate ALCE scores of the candidate it chose
T = {}
for l in open(f"{R}/combined/cand_metrics_rr_C_exact.jsonl"):
    r = json.loads(l); T[(int(r["q_idx"]), r["p_idx"])] = r
fin = {int(x["q_idx"]): x for x in json.load(open(f"{R}/combined/final_948.json"))["data"]}
base = {q: T[(q, int(fin[q]["chosen_p_idx"]))] for q in S if (q, int(fin[q].get("chosen_p_idx", -1))) in T}
qs = sorted(base)
print(f"Set C questions with a single-segment combined answer: {len(qs)}")
print("combined system:", {k: round(st.mean(base[q][k] for q in qs), 2) for k in K})
rows = (("New passage, checked", "seg2_filter_948"), ("New passage, own choice", "seg2_beam_948"),
        ("Same passage, checked", "seg2_same_filter_948"), ("Same passage, own choice", "seg2_same_beam_948"))
P = {n: perq(n) for _, n in rows}
print("\nDifference from the combined system:")
for lab, n in rows:
    print(f"  {lab:25s}" + " | ".join(f"{k} {ci([P[n][q][k] - base[q][k] for q in qs])}" for k in K[:3])
          + f" | words {st.mean(P[n][q]['length'] - base[q]['length'] for q in qs):+.1f}")
print("\nNew passage minus same passage, model's own choice (matched length):")
a, b = P["seg2_beam_948"], P["seg2_same_beam_948"]; q2 = [q for q in S if q in a and q in b]
print(f"  n={len(q2)}  " + " | ".join(f"{k} {ci([a[q][k] - b[q][k] for q in q2])}" for k in ("str_em", "citation_rec", "length")))
