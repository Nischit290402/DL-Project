# Self-contradiction across candidates

Branch `self-contradiction` (from `critique-p1`, reuses its saved candidates, test sets and scorers).

**Why candidates, not segments.** The original idea penalised contradictions between segments of one answer during
beam search. On ASQA that cannot be tested: C0 has no answer with two or more segments and C1 has one
(`results/critique_p1.md`, contradiction rate row). Self-RAG does write 5 candidate answers per question, one per
retrieved passage, and ranks each in isolation (relevance + support). Hypothesis: a candidate that the other
candidates contradict is more likely wrong.

**Step 1: go/no-go diagnostics on Set A** (Set B stays untouched until the method is fixed).

| | Question | Stop rule |
| --- | --- | --- |
| D1 | How often do candidates of the same question contradict each other? | if almost never, nothing to penalise |
| D2 | Within a question, is the more-contradicted candidate less correct (str-em)? | if the 95% interval of the concordance includes 0.5, stop and report a negative result |
| D3 | Of 30 random flagged pairs, how many are real contradictions vs different readings of an ambiguous question? | reported whatever it shows |

C_i = mean over the other candidates j of max(p_con(i->j), p_con(j->i)), with `cross-encoder/nli-deberta-v3-large`
(not the T5-11B model ALCE grades citations with). Two variants: plain pairs (`noctx`) and with the question in front
of the premise (`ctx`).

```bash
# cluster (GPU, ~5 min)
cd ~/DL-Project && git fetch && git checkout self-contradiction && git pull
sbatch cluster/run_cand_nli.sbatch                       # writes results/runs/cand_nli_A.jsonl
# laptop (stdlib only), after copying the jsonl back:
scp "siyuan029@10.96.189.12:~/DL-Project/results/runs/cand_nli_A.jsonl" results/runs/
python3 contradiction/diagnostics.py --nli results/runs/cand_nli_A.jsonl \
  --f results/runs/asqa_cands_0_end.json --split critique/splits/asqa_A.json --out results/cand_contra_A
```

Outputs: `results/cand_contra_A.json` (D1, D2 for both variants) and `results/cand_contra_A_pairs.csv` (D3: all
pairs with p_con >= 0.5 under `ctx`, 30 random ones marked `sample30=1`; fill `label` with `genuine`,
`different-reading` or `nli-error`).

Reading D2: concordance = among candidate pairs of the same question with different str-em, the share where the
better one has the lower score. 0.5 = no signal. The same number is given for Self-RAG's released score and for a
word-overlap agreement score, so the NLI signal can be judged against both.
