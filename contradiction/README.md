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

## Step 1 results (Set A, 2026-11-17)

Full numbers: `results/cand_contra_A.json`. Labels: `results/cand_contra_A_labels.csv` (suggested labels; the
`checked_by_human` column is filled in once reviewed).

- **D1.** 88% of questions (89% with question context) have a candidate pair with p_con >= 0.5, and 56% of all
  pairs are flagged. Raising the threshold to 0.9 barely changes this: the NLI model is close to binary.
- **D2.** Concordance of C: 0.586 [0.503, 0.669] (noctx), 0.607 [0.514, 0.688] (ctx); Self-RAG's own score
  0.682 [0.609, 0.753]; word overlap 0.588. str-em falls with C (33.9 / 27.2 / 22.1 / 17.1 by bucket). In the
  19-22 questions where the chosen answer is the most contradicted, the least contradicted alternative is worse.
  Passes the stop rule narrowly; weaker than Self-RAG's own ranking.
- **D3.** 30 random flagged pairs (ctx): 11 genuine, 13 different-reading, 6 nli-error. In 6 of the 11 genuine
  pairs both answers are wrong. 3 of the 4 pairs flagged only with question context are NLI errors.

## Step 2: settings fixed on Set A before touching Set B (2026-11-17)

Sweep of new score = final_score - lambda * C on Set A (str-em; released 27.78):

| Variant | lambda 0.05 | 0.1 | 0.2 | 0.3 | 0.5 | 0.75 | 1 | 1.5 | 2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| noctx | 27.62 | 27.62 | 28.30 | **28.50** | 28.25 | 27.75 | 27.42 | 26.75 | 26.42 |
| ctx | 27.22 | 27.72 | 27.67 | 28.07 | 28.72 | 27.38 | 27.38 | 26.72 | 24.72 |

An agreement bonus (+ mu * E) never beat 28.22. **Chosen: noctx, lambda = 0.3** (middle of the 0.2-0.5 plateau;
the ctx peak is a single point and D3 shows context adds false alarms). On Set A: +0.72 [-2.00, +3.25], 21 of 100
answers changed (6 better, 4 worse). These are the defaults in `contradiction/rerank_contra.py`; `--lam 0`
reproduces `results/runs/rr_released.json` on all 100 Set A questions.

## Step 3: one held-out test on Set B

```bash
# cluster: NLI on Set B, re-pick, per-question scores, in one job (~20-30 min)
sbatch cluster/run_contra_eval.sbatch
# laptop, after copying results/runs/rr_contra_B.json.perq.jsonl back:
python3 critique/bootstrap.py --base results/runs/rr_released.json.perq.jsonl \
  --new results/runs/rr_contra_B.json.perq.jsonl --split critique/splits/asqa_B.json
```
