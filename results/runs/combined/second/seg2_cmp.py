import json, random, statistics as st, os, sys
K=("str_em","rougeLsum","citation_rec","citation_prec","length")
def ci(v,B=2000):
    r=random.Random(0); n=len(v); ms=sorted(st.mean(r.choices(v,k=n)) for _ in range(B)); return "%+.2f [%+.2f, %+.2f]"%(st.mean(v),ms[int(.025*B)],ms[int(.975*B)])
S=json.load(open(sys.argv[1]))["indices"]; T={}
for s in ("AB","C"):
    for l in open(f"results/runs/retrieval/cand_metrics_rr_{s}_exact.jsonl"): r=json.loads(l); T[(int(r["q_idx"]),r["p_idx"])]=r
fin={int(x["q_idx"]):x for x in json.load(open("results/runs/combined/final_948.json"))["data"]}
base={q:T[(q,int(fin[q]["chosen_p_idx"]))] for q in S if (q,int(fin[q].get("chosen_p_idx",-1))) in T}
old={int(json.loads(l)["q_idx"]):json.loads(l) for l in open("results/runs/retrieval/stack/final_two_gate50_948.json.perq.jsonl")}
qs=sorted(base); print("n=",len(qs)); print("one passage (final)     ",{k:round(st.mean(base[q][k] for q in qs),2) for k in K})
print("append existing, filter ",{k:round(st.mean(old[q][k] for q in qs),2) for k in K})
for p in sys.argv[2:]:
    if not os.path.exists(p+".summary.json"): print(p,"pending"); continue
    s=json.load(open(p+".summary.json")); pq={int(json.loads(l)["q_idx"]):json.loads(l) for l in open(p+".perq.jsonl")}; q2=[q for q in qs if q in pq]
    print(p,{k:round(st.mean(pq[q][k] for q in q2),2) for k in K},"MAUVE(file)",round(s.get("mauve",0),2),"n",len(q2))
    print("   vs one passage:    "," | ".join(k+" "+ci([pq[q][k]-base[q][k] for q in q2]) for k in K))
    print("   vs append existing:"," | ".join(k+" "+ci([pq[q][k]-old[q][k] for q in q2]) for k in K))
