"""Exploratory (not pre-registered): which candidate should be picked? Compares Self-RAG's own pick with
'take the candidate written from the first-ranked passage'. Prints only the split given."""
import json, re, string, random, statistics as st, sys
_P=set(string.punctuation)
def norm(s):
    s=s.lower(); s="".join(ch for ch in s if ch not in _P); s=re.sub(r"\b(a|an|the)\b"," ",s); return " ".join(s.split())
def strem(t,qa):
    t=norm(t); return 100*st.mean([1.0 if any(norm(a) in t for a in q['short_answers']) else 0.0 for q in qa]) if qa else 0.0
TOK=r"\[(Retrieval|No Retrieval|Continue to Use Evidence|Relevant|Irrelevant|Fully supported|Partially supported|No support / Contradictory|Utility:[1-5])\]"
clean=lambda p: re.sub(TOK," ",p).replace("</s>"," ")
def ci(v,B=2000):
    r=random.Random(0); n=len(v); ms=sorted(st.mean(r.choices(v,k=n)) for _ in range(B)); return st.mean(v),ms[int(.025*B)],ms[int(.975*B)]
def rows(path):
    R={}
    for x in json.load(open(path))['data']:
        c=sorted([c for c in x['candidates'] if c['depth']==1],key=lambda c:c['p_idx']); qa=x['qa_pairs']
        top=max(c,key=lambda k:k['beam_score'])
        em=[strem(clean(k['pred']),qa) for k in c]; w=[len(clean(k['pred']).split()) for k in c]
        R[x['q_idx']]=dict(own=em[top['p_idx']],first=em[0],own_w=w[top['p_idx']],first_w=w[0],same=int(top['p_idx']==0))
    return R
ref=rows(sys.argv[1]); new=rows(sys.argv[2]); S=json.load(open(sys.argv[3]))['indices']; S=[i for i in S if i in ref and i in new]
f=lambda lab,v: print("  %-58s %6.2f [%6.2f,%6.2f]"%((lab,)+ci(v)))
print("n =",len(S))
f("C1: Self-RAG's pick (GTR top-5)",[ref[i]['own'] for i in S])
f("GTR top-5, candidate from GTR rank-1 passage",[ref[i]['first'] for i in S])
f("Reranked top-5, Self-RAG's pick",[new[i]['own'] for i in S])
f("Reranked top-5, candidate from reranker's rank-1 passage",[new[i]['first'] for i in S])
f("  diff: rerank+first  minus C1",[new[i]['first']-ref[i]['own'] for i in S])
f("  diff: rerank+first  minus rerank+own pick",[new[i]['first']-new[i]['own'] for i in S])
f("  diff: GTR first     minus C1",[ref[i]['first']-ref[i]['own'] for i in S])
print("  Self-RAG's pick is the first-ranked passage in: GTR %d/%d, reranked %d/%d"%(sum(ref[i]['same'] for i in S),len(S),sum(new[i]['same'] for i in S),len(S)))
print("  words: C1 %.1f | rerank+first %.1f"%(st.mean(ref[i]['own_w'] for i in S),st.mean(new[i]['first_w'] for i in S)))
