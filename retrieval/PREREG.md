> Note added when this file entered git (21 Nov 2026, after the runs): the text below the line was saved in the
> project's notes at 14:17 SGT on 21 Nov 2026, before the first reranker job (56491, submitted 14:24). It is copied
> here unchanged. Git history therefore does not show the order; the cluster records do: `retrieval/records/`
> and the SLURM job times. `division1-proving-results.md` is now `results/retrieval.md`, Part 1.

---

# Division 1: pre-registration of the reranking experiment

Written 21 Nov 2026, about 14:30 SGT, **before any reranker has been run on any ASQA question**. This document's creation time in the project is the timestamp. Nothing below is edited after results exist; changes go into a "Deviations" section at the end with a date and a reason.

## 1. Why this experiment (evidence already collected, see `division1-proving-results.md`)

| Finding | Number | What it implies |
|---|---|---|
| Better passages raise the end metric at no length cost | Oracle top-5: str-em +9.2 [6.0, 12.6], Set A+B | Passage choice is a real lever |
| Self-RAG cannot find better passages itself | 10 passages: -0.2 [-2.7, 2.3]; 20 passages: -1.0 [-4.7, 2.6] | The ranking signal must come from outside the generator |
| The pool has more to offer | Best single passage: 51.2 in top 5, 59.1 in top 20, 63.8 in top 100 (948 questions) | A reranker has room to work |
| The answer keeps about 85% of what its passage holds | str-em / coverage of chosen passage = 0.83 to 0.88 in four runs | Coverage of the passage used should predict str-em |

Gap in the released system: the 5 passages are the GTR top 5, taken once (`run_long_form_static.py:482`), and GTR is a bi-encoder that scores question and passage separately.

## 2. Method under test

Rescore the GTR top-K passages of each question with a cross-encoder reranker (question and passage read together), keep the 5 highest, and run Self-RAG with C1 settings on those 5. Nothing inside Self-RAG changes. The reranker sees only the question and the passages, never gold answers.

Configurations allowed (4, fixed now):

| | Reranker | Pool depth K |
|---|---|---|
| R1-20, R1-100 | `cross-encoder/ms-marco-MiniLM-L-12-v2` | 20, 100 |
| R2-20, R2-100 | `BAAI/bge-reranker-v2-m3` | 20, 100 |

Passage text given to the reranker: `title + " " + text`, max 512 tokens. No other reranker, K, or text format will be tried.

## 3. Metrics

Component (retrieval stage), computed from passage text and ASQA gold short answers with the matcher already validated against ALCE (15,920 of 15,920 labels):
- **mean@5 (primary):** mean answer coverage of the 5 passages, one at a time. Chosen as primary because Self-RAG writes from one passage and its pick is close to uninformed (coverage of its pick 36.4 vs rank-1 passage 35.7).
- rank1, best1@5, union@5 (secondary).

End task: str-em (primary), answer length in words, seconds per question including reranking. ROUGE-L, MAUVE and citation recall/precision are reported if the T5 scorer is available; they are not part of any decision rule.

## 4. Hypotheses and decision rules

**Step 1, configuration choice, Set A only, no Self-RAG run.** Pick the configuration with the highest mean@5 on Set A. Ties (within 0.5 points) go to the smaller model, then the smaller K.

**Gate G1 (component).** On Set A, the chosen configuration's mean@5 minus GTR's, paired bootstrap over questions, must have a 95% interval above 0. If not: stop, report that off-the-shelf rerankers do not raise answer coverage on ASQA, run nothing further.

**H1 (component, Set B).** mean@5 of the chosen configuration is higher than GTR's on Set B, interval above 0.

**H2 (end task, Set B).** One Self-RAG run on Set A+B with the chosen configuration. str-em on Set B is higher than C1's (paired). Answer length does not rise by more than 3 words.

**Quantitative prediction, to test the 85% relation.** Let d = (coverage of the passage Self-RAG picks, reranked) minus (the same, C1). Predicted str-em change = 0.85 x d. Report predicted and observed side by side on Set B.

**Step 2, Set C.** Only if the Set B str-em difference is positive (point estimate). Then exactly one run on the 748 Set C questions with the same configuration, reported whatever it shows. Confirmed if the Set C str-em interval is above 0.

Outcomes and their meaning:
- H1 yes, H2 yes: full chain, stage metric and end metric move together.
- H1 yes, H2 no: better passages were offered and Self-RAG did not use them. Division 1's result then becomes evidence for the picking problem that Divisions 2 and 3 work on.
- G1 fails: relevance rerankers do not select for answer content on ASQA.

## 5. Expectations written before running

1. The reranker raises rank1 and mean@5 more than it raises union@5.
2. union@5 may fall. ASQA questions are ambiguous and need passages for several readings; a relevance reranker may return five passages about the same reading.
3. The str-em gain will be well below the oracle's +9.2, because the oracle used gold answers.
4. K = 20 may beat K = 100: the deeper pool has higher coverage but more near-miss passages.

## 6. Fixed settings

Self-RAG 7B, team branch `critique-p1` unmodified, `cluster/run_asqa.sbatch`, flags `--keep_special_tokens --save_candidates --ndocs 5` plus the README long-form flags, NVIDIA L40S only. Baseline: C1 (team's saved candidates for Set A+B, reproduced 200 of 200 from this account on 17 Nov; team's `asqa_cands_C.json` for Set C). Bootstrap: paired over questions, 2,000 resamples, seed 0.

## 7. Prior use of the data

- Set A and Set B: used by every division for tuning and testing. For Division 1, all runs so far (control, oracle, 10 and 20 passages, stitching) were on Set A+B.
- Set C: used by Division 4 and Division 2 for their own hypotheses. Division 1 has computed only passage coverage on all 948 questions (P1), which includes Set C; no Self-RAG output on Set C has been looked at for this division.

## 8. Controls already run (no reranker)

GTR top-5 (C1), oracle top-5 (ceiling), GTR top-10 and top-20 with Self-RAG's own scoring (wider pool, no reranker).

## Deviations

(none yet)
