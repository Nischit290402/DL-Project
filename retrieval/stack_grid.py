"""Level 2 grid: passages (GTR top 5 / reranked top 5) x picking rule (C1 / C2 / D2-v2), all from saved candidates.

Picking rules are the other divisions' own code and frozen settings, unchanged:
  C1     critique/learn_weights.py select(), results/critique_paper_scoring_configs/released.json
  C2     same, paper_intended.json (utility and sequence terms restored, paper weights)
  D2-v2  grounding/rerank_ground.py, parameters of grounding/configs/d2v2_final.json
Per-question metrics are looked up in the exact per-candidate tables (ALCE scores with T5-11B), as
grounding/lookup_perq.py does. Multi-segment answers are not in those tables and are left out of every row.

usage (repo root): python3 retrieval/stack_grid.py [--bases GTR,RR] [--out results/runs/retrieval/stack]"""
import argparse, json, os, random, statistics as st, subprocess, sys
sys.path.insert(0, "critique")
import learn_weights as LW

R, RR = "results/runs", "results/runs/retrieval"
BASES = {
 "GTR": {"AB": dict(cands=f"{R}/asqa_cands_0_end.json", metrics=f"{R}/cand_metrics_AB_exact.jsonl",
                    ground=f"{R}/cand_ground_A.jsonl,{R}/cand_ground_B.jsonl", rel=f"{R}/cand_qarel_AB.jsonl",
                    split="critique/splits/asqa_A.json,critique/splits/asqa_B.json"),
         "C":  dict(cands=f"{R}/asqa_cands_C.json", metrics=f"{R}/cand_metrics_C_exact.jsonl",
                    ground=f"{R}/cand_ground_C.jsonl", rel=f"{R}/cand_qarel_C.jsonl", split="critique/splits/asqa_C.json")},
 "RR":  {"AB": dict(cands=f"{RR}/asqa_rrR2K20_0_end.json", metrics=f"{RR}/cand_metrics_rr_AB_exact.jsonl",
                    ground=f"{RR}/cand_ground_rr_AB.jsonl", rel=f"{RR}/cand_qarel_rr_AB.jsonl",
                    split="critique/splits/asqa_A.json,critique/splits/asqa_B.json"),
         "C":  dict(cands=f"{RR}/asqa_rrR2K20C_0_end.json", metrics=f"{RR}/cand_metrics_rr_C_exact.jsonl",
                    ground=f"{RR}/cand_ground_rr_C.jsonl", rel=f"{RR}/cand_qarel_rr_C.jsonl", split="critique/splits/asqa_C.json")},
}
KEYS = ("str_em", "rougeLsum", "citation_rec", "citation_prec", "length")
D2V2 = json.load(open("grounding/configs/d2v2_final.json"))
CFG = {k: json.load(open(f"results/critique_paper_scoring_configs/{f}.json")) for k, f in (("C1", "released"), ("C2", "paper_intended"))}

def ci(v, B=2000):
    r = random.Random(0); n = len(v); ms = sorted(st.mean(r.choices(v, k=n)) for _ in range(B))
    return st.mean(v), ms[int(.025 * B)], ms[int(.975 * B)]

def picks_for(base, s, out_dir):
    f = BASES[base][s]; data = json.load(open(f["cands"]))["data"]
    table = {}
    for line in open(f["metrics"]):
        r = json.loads(line); table[(r["q_idx"], r["node_id"])] = r
    out = f"{out_dir}/rr_{base}_{s}_d2v2.json"
    cmd = [sys.executable, "grounding/rerank_ground.py", "--f", f["cands"], "--ground", f["ground"], "--split", f["split"],
           "--mode", D2V2["mode"], "--agg", D2V2["agg"], "--param", str(D2V2["param"]), "--rel", f["rel"],
           "--rel_model", D2V2["gate"]["rel_model"], "--delta", str(D2V2["gate"]["delta"]), "--out", out]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    d2 = {it["q_idx"]: it.get("chosen_node_id") for it in json.load(open(out))["data"]}
    rows = {}
    for item in data:
        q = item["q_idx"]; path = item["best_paths"][str(item.get("output_path", 0))]
        if sum(node not in (None, 0) for node in path) != 1: continue            # multi-segment answer
        cands = LW.first_step(item); node = {}
        for k in ("C1", "C2"):
            feats = [(c, LW.features(c, CFG[k]["temperature"])) for c in cands]
            node[k] = LW.select(feats, CFG[k]["weights"])["node_id"]
        node["D2v2"] = d2.get(q)
        if any((q, n) not in table for n in node.values()): continue
        rows[q] = {k: dict({m: table[(q, n)][m] for m in KEYS}, node=n) for k, n in node.items()}
        rows[q]["_sel"] = next((r["node_id"] for (qq, n), r in table.items() if qq == q and r.get("selected")), None)
    return rows

ap = argparse.ArgumentParser(); ap.add_argument("--bases", default="GTR,RR"); ap.add_argument("--out", default=f"{RR}/stack")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); bases = a.bases.split(",")
G = {}                                                     # G[(base, rule)][q] = metrics
for b in bases:
    for s in ("AB", "C"):
        rows = picks_for(b, s, a.out)
        same = sum(r["C1"]["node"] == r["_sel"] for r in rows.values())
        print(f"check {b} {s}: C1 rule equals the run's own choice on {same}/{len(rows)} questions")
        for q, r in rows.items():
            for k in ("C1", "C2", "D2v2"): G.setdefault((b, k), {})[q] = r[k]
if "GTR" in bases and os.path.exists(f"{R}/rr_ground_v2_C.json"):
    ref = {it["q_idx"]: it.get("chosen_node_id") for it in json.load(open(f"{R}/rr_ground_v2_C.json"))["data"]}
    mine = G[("GTR", "D2v2")]; common = [q for q in ref if q in mine]
    print(f"check GTR D2-v2: same pick as the grounding division's committed Set C file on {sum(mine[q]['node'] == ref[q] for q in common)}/{len(common)}")
common = set.intersection(*[set(v) for v in G.values()])
C = set(json.load(open("critique/splits/asqa_C.json"))["indices"])
for (b, k), rows in G.items():
    with open(f"{a.out}/{b}_{k}.perq.jsonl", "w") as f:
        for q in sorted(common): f.write(json.dumps(dict(q_idx=q, **{m: rows[q][m] for m in KEYS})) + "\n")
order = [(b, k) for b in bases for k in ("C1", "C2", "D2v2")]
for name, idx in (("Set C", sorted(common & C)), ("Set A+B", sorted(common - C)), ("All", sorted(common))):
    print(f"\n=== {name} (n={len(idx)})\n  {'row':10s} " + "  ".join(f"{m:>13s}" for m in KEYS))
    for b, k in order:
        print(f"  {b+'+'+k:10s} " + "  ".join(f"{st.mean(G[(b,k)][q][m] for q in idx):13.2f}" for m in KEYS))
    for ref in [r for r in (("GTR", "C2"), ("GTR", "C1")) if r in G]:
        print(f"  -- paired difference from {ref[0]}+{ref[1]}")
        for b, k in order:
            if (b, k) == ref: continue
            print(f"  {b+'+'+k:10s} " + "  ".join("%+5.2f[%+5.2f,%+5.2f]" % ci([G[(b,k)][q][m] - G[ref][q][m] for q in idx]) for m in KEYS))
