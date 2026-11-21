"""Answer coverage of the retrieved passages (no GPU, standard library only).
For each ASQA question and each cutoff k: share of gold sub-questions (qa_pairs) with at least one short answer
found as a normalised substring in (union) the first k passages together, (best1) the best single passage among them.
Matching follows ALCE: utils.normalize_answer + substring (compute_str_em)."""
import json, re, string, sys, random, statistics as st
KS=[1,2,3,5,10,20,50,100]
_P=set(string.punctuation)
def norm(s):
    s=s.lower(); s="".join(ch for ch in s if ch not in _P)
    s=re.sub(r"\b(a|an|the)\b"," ",s); return " ".join(s.split())
def ci(v,B=1000,seed=0):
    r=random.Random(seed); n=len(v); ms=sorted(st.mean(r.choices(v,k=n)) for _ in range(B))
    return (st.mean(v),ms[int(.025*B)],ms[int(.975*B)])
inp,out=sys.argv[1],sys.argv[2]; splits=sys.argv[3:]
data=json.load(open(inp)); print("questions:",len(data),"| doc keys:",sorted(data[0]["docs"][0].keys()),"| ndocs/q min,max:",min(len(x["docs"]) for x in data),max(len(x["docs"]) for x in data))
srt=0; has_af="answers_found" in data[0]["docs"][0]; af_agree=af_tot=0
rows=[]
for qi,x in enumerate(data):
    qa=[[norm(a) for a in q["short_answers"]] for q in x["qa_pairs"]]
    docs=x["docs"]
    try:
        sc=[float(d["score"]) for d in docs]; srt+= all(sc[i]>=sc[i+1]-1e-9 for i in range(len(sc)-1))
    except Exception: pass
    hit={}
    for mode in ("text","title_text"):
        H=[]
        for d in docs:
            t=norm(d["text"] if mode=="text" else d.get("title","")+" "+d["text"])
            H.append([any(a in t for a in sa) for sa in qa])
        hit[mode]=H
    if has_af:
        for d,h in zip(docs,hit["text"]):
            af=d["answers_found"]
            if isinstance(af,str): af=json.loads(af)
            if len(af)==len(h): af_tot+=len(h); af_agree+=sum(int(bool(a))==int(b) for a,b in zip(af,h))
    r={"q_idx":qi,"n_qa":len(qa)}
    for mode,H in hit.items():
        for k in KS:
            Hk=H[:k]
            if not qa or not Hk: r[f"{mode}_union@{k}"]=0.0; r[f"{mode}_best1@{k}"]=0.0; continue
            r[f"{mode}_union@{k}"]=100*st.mean(any(h[j] for h in Hk) for j in range(len(qa)))
            r[f"{mode}_best1@{k}"]=100*max(st.mean(h) for h in Hk)
    rows.append(r)
with open(out,"w") as f:
    for r in rows: f.write(json.dumps(r)+"\n")
print("questions with passages sorted by score (desc):",srt,"/",len(data))
if has_af: print("agreement of my matcher with ALCE's per-doc answers_found: %d/%d = %.4f"%(af_agree,af_tot,af_agree/max(af_tot,1)))
groups={"ALL":list(range(len(data)))}
for s in splits:
    j=json.load(open(s)); groups[j.get("name",s)]=j["indices"]
if len(splits)>=2: groups["A+B"]=sorted(set(sum((json.load(open(s))["indices"] for s in splits),[])))
for g,idx in groups.items():
    print(f"\n== {g} (n={len(idx)})  mean [95% CI]")
    for mode in ("text","title_text"):
        for kind in ("union","best1"):
            line=f"{mode:10s} {kind:5s} "
            for k in KS:
                m,l,h=ci([rows[i][f"{mode}_{kind}@{k}"] for i in idx]); line+=f"@{k}: {m:5.1f} [{l:4.1f},{h:4.1f}]  "
            print(line)
    for a,b,lab in (("title_text_union@100","title_text_union@5","union@100 - union@5"),("title_text_best1@100","title_text_best1@5","best1@100 - best1@5"),("title_text_union@5","title_text_best1@5","union@5 - best1@5"),("title_text_best1@5","title_text_best1@1","best1@5 - rank1"),("title_text_best1@20","title_text_best1@5","best1@20 - best1@5"),("title_text_best1@10","title_text_best1@5","best1@10 - best1@5")):
        m,l,h=ci([rows[i][a]-rows[i][b] for i in idx]); print(f"  paired gap {lab:22s} {m:+5.1f} [{l:+5.1f},{h:+5.1f}]")
    z=lambda key: sum(rows[i][key]==0 for i in idx)
    print(f"  questions with zero coverage: union@5 {z('title_text_union@5')}, union@100 {z('title_text_union@100')}")
