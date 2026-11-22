"""Offline bound on multi-passage answers. Stitch the top-k C1 candidates (by Self-RAG's own score) into one
answer, no new generation. Reports str-em (ALCE definition), length in words, and passage coverage."""
import json, re, string, random, statistics as st, sys
_P=set(string.punctuation)
def norm(s):
    s=s.lower(); s="".join(ch for ch in s if ch not in _P); s=re.sub(r"\b(a|an|the)\b"," ",s); return " ".join(s.split())
def strem(text, qa):
    t=norm(text); return 100*st.mean([1.0 if any(norm(a) in t for a in q['short_answers']) else 0.0 for q in qa]) if qa else 0.0
TOK=r"\[(Retrieval|No Retrieval|Relevant|Irrelevant|Fully supported|Partially supported|No support / Contradictory|Utility:[1-5])\]"
def sentences(pred):
    p=re.sub(TOK,"",pred).replace("</s>","")
    return [s.strip() for s in p.split("[Continue to Use Evidence]") if s.strip()]
def stitch(cands, k, order="score"):
    d1=[c for c in cands if c['depth']==1]
    d1=sorted(d1,key=lambda c:-c['beam_score']) if order=="score" else sorted(d1,key=lambda c:c['p_idx'])
    out=[]; docs=[]; seen=[]
    for c in d1:
        if len(docs)==k: break
        ss=[s for s in sentences(c['pred']) if norm(s) and not any(norm(s) in x or x in norm(s) for x in seen)]
        if not ss: continue
        docs.append(c['ctx']); seen+= [norm(s) for s in ss]
        for s in ss:
            s=s[:-1] if s[-1] in ".!?" else s
            out.append(f"{s} [{len(docs)}].")
    return " ".join(out), docs
def ci(v,B=2000):
    r=random.Random(0); n=len(v); ms=sorted(st.mean(r.choices(v,k=n)) for _ in range(B)); return st.mean(v),ms[int(.025*B)],ms[int(.975*B)]
data=json.load(open(sys.argv[1]))['data']
A=set(json.load(open('critique/splits/asqa_A.json'))['indices']); B=set(json.load(open('critique/splits/asqa_B.json'))['indices'])
res={}; outs={}
for k in (1,2,3,5):
    rows=[]; items=[]
    for x in data:
        text,docs=stitch(x['candidates'],k)
        plain=re.sub(r"\[\d+\]","",text)
        cov=strem(" ".join(d['title']+" "+d['text'] for d in docs),x['qa_pairs'])
        rows.append(dict(q=x['q_idx'],em=strem(plain,x['qa_pairs']),len=len(plain.split()),nd=len(docs),cov=cov))
        y={kk:vv for kk,vv in x.items() if kk not in('candidates','best_paths','output_path','intermediate')}; y['output']=text; y['docs']=docs; items.append(y)
    res[k]=rows; outs[k]=items
ref=[strem(re.sub(r"\[\d+\]","",x['output']),x['qa_pairs']) for x in data]
print("check: stitch k=1 equals the C1 chosen answer str-em on %d/%d questions; C1 mean %.2f"%(sum(abs(a-r['em'])<1e-9 for a,r in zip(ref,res[1])),len(ref),st.mean(ref)))
for name,S in (("A",A),("B",B),("A+B",A|B)):
    print(f"\n== Set {name}")
    base=[r for r in res[1] if r['q'] in S]
    for k in (1,2,3,5):
        rows=[r for r in res[k] if r['q'] in S]
        m,l,h=ci([r['em'] for r in rows]); dm,dl,dh=ci([r['em']-b['em'] for r,b in zip(rows,base)])
        print(f" top-{k}: str-em {m:5.2f} [{l:5.2f},{h:5.2f}]  vs top-1 {dm:+5.2f} [{dl:+5.2f},{dh:+5.2f}]  words {st.mean(r['len'] for r in rows):5.1f}  passages used {st.mean(r['nd'] for r in rows):.2f}  coverage of passages used {st.mean(r['cov'] for r in rows):5.1f}  em/coverage {st.mean(r['em'] for r in rows)/st.mean(r['cov'] for r in rows):.2f}")
for k in (2,3):
    json.dump({'data':outs[k]},open(sys.argv[2]+f"_top{k}.json",'w'))
print("\nexample (top-2):",outs[2][0]['output'][:400])
