"""Compare runs made with --save_candidates against a C1 reference file on the same questions. Standard library only.
usage: analyze_runs.py REF_CANDS.json RUN1.json [RUN2.json ...]"""
import json, re, string, random, statistics as st, sys, os
_P=set(string.punctuation)
def norm(s):
    s=s.lower(); s="".join(ch for ch in s if ch not in _P); s=re.sub(r"\b(a|an|the)\b"," ",s); return " ".join(s.split())
def strem(text, qa):
    t=norm(text); return 100*st.mean([1.0 if any(norm(a) in t for a in q['short_answers']) else 0.0 for q in qa]) if qa else 0.0
TOK=r"\[(Retrieval|No Retrieval|Continue to Use Evidence|Relevant|Irrelevant|Fully supported|Partially supported|No support / Contradictory|Utility:[1-5])\]"
clean=lambda p: re.sub(TOK," ",p).replace("</s>"," ")
plain=lambda o: re.sub(r"\[\d+\]","",o)
def ci(v,B=2000):
    r=random.Random(0); n=len(v); ms=sorted(st.mean(r.choices(v,k=n)) for _ in range(B)); return st.mean(v),ms[int(.025*B)],ms[int(.975*B)]
def rows_of(path):
    d=json.load(open(path)); R={}
    for x in d['data']:
        qa=x['qa_pairs']; c1=[c for c in x['candidates'] if c['depth']==1]
        top=max(c1,key=lambda c:c['beam_score'])
        cov=[strem(c['ctx']['title']+" "+c['ctx']['text'],qa) for c in c1]
        em=[strem(clean(c['pred']),qa) for c in c1]
        R[x['q_idx']]=dict(out=x['output'],em=strem(plain(x['output']),qa),words=len(plain(x['output']).split()),
            sel_oracle=max(em),cand_mean=st.mean(em),cov_chosen=cov[c1.index(top)],cov_best1=max(cov),
            cov_union=strem(" ".join(c['ctx']['title']+" "+c['ctx']['text'] for c in c1),qa),
            chosen_p=top['p_idx'],ncand=len(c1),nseg=len(x['intermediate'])-1,maxdepth=max(c['depth'] for c in x['candidates']),
            sup_full=sum('[Fully supported]' in c['pred'] for c in c1)/len(c1))
    return R, d.get('args',{}), d.get('total_cost')
ref,_,_=rows_of(sys.argv[1])
HERE=os.path.dirname(os.path.abspath(sys.argv[0]))
A=set(json.load(open(os.path.join(HERE,'..','critique','splits','asqa_A.json')))['indices'])
print("REF = team C1 saved candidates, n=%d"%len(ref))
for path in sys.argv[2:]:
    R,args,cost=rows_of(path); ids=sorted(set(R)&set(ref))
    print(f"\n=== {os.path.basename(path)}  n={len(ids)}  ndocs={args.get('ndocs')} input={os.path.basename(str(args.get('input_file')))}")
    same=sum(R[i]['out']==ref[i]['out'] for i in ids); print(f"  answers identical to REF: {same}/{len(ids)} | candidates/question: {st.mean(R[i]['ncand'] for i in ids):.1f} | answers with >1 segment: {sum(R[i]['nseg']>1 for i in ids)} | max tree depth: {max(R[i]['maxdepth'] for i in ids)}")
    inA=[i for i in ids if i in A]; parts=[(ids,"all questions in this run")]+([(inA,"Set A"),([i for i in ids if i not in A],"Set B")] if 0<len(inA)<len(ids) else [])
    for S,name in parts:
        print(f"  -- {name} (n={len(S)})")
        for key,lab in (("em","answer str-em"),("sel_oracle","best candidate str-em (selection oracle)"),("cand_mean","mean candidate str-em"),("cov_chosen","coverage: passage chosen"),("cov_best1","coverage: best single passage offered"),("cov_union","coverage: all passages offered"),("words","answer length (words)")):
            m,l,h=ci([R[i][key] for i in S]); dm,dl,dh=ci([R[i][key]-ref[i][key] for i in S])
            print(f"     {lab:42s} {m:6.2f} [{l:6.2f},{h:6.2f}]   vs REF {dm:+6.2f} [{dl:+6.2f},{dh:+6.2f}]")
        if name.startswith("all"):
            m=st.mean(R[i]['em'] for i in S); c=st.mean(R[i]['cov_chosen'] for i in S)
            hist={}; 
            for i in S: b=min(R[i]['chosen_p']//5*5,95); hist[b]=hist.get(b,0)+1
            print(f"     em / coverage of chosen passage = {m/c:.2f} | rank bucket of chosen passage (0 = ranks 1-5): {sorted(hist.items())} | share of candidates graded [Fully supported]: {st.mean(R[i]['sup_full'] for i in S):.2f}")
