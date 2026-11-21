# Retrieval division: does passage choice limit Self-RAG on ASQA, and can a better choice help?

All runs: Self-RAG 7B, C1 settings (`--keep_special_tokens`, README long-form flags), NTU TC2, NVIDIA L40S,
`self-rag/` as on `critique-p1`. Intervals are 95% bootstrap over questions (2,000 resamples, seed 0);
differences are paired on the same questions. Commands: `retrieval/README.md`.

**Coverage** of a passage (or an answer) = share of the question's gold sub-answers (ASQA `qa_pairs`) with a
short answer found as a normalised substring. This is ALCE's EM recall, applied to passages.

## Summary

1. The released ASQA setup gives every question the GTR top 5 once. No retriever runs, and 947 of 948 answers
   are one segment written from one passage.
2. Gold answers are lost in a chain (Set A+B): 59.5% are in the 5 passages, 53.1% in the best single one,
   36.4% in the one Self-RAG picks, 30.9% in the answer.
3. Oracle passages raise str-em by +9.2 [6.0, 12.6] at the same length. Showing Self-RAG 10 or 20 passages
   does not (-0.2, -1.0): its own scoring does not find the better ones.
4. A cross-encoder reranker (GTR top 20 reordered, top 5 kept), chosen and tested under a plan fixed in
   advance, raises the coverage of the 5 passages by +3.1 [2.4, 3.8] and str-em by **+1.7 [0.3, 3.1]** on
   748 held-out questions, at 0.17 s per question and no change in length.

## Part 1. Diagnosis (17 Nov 2026)

### Checks made before trusting any number

| Check | Result |
| --- | --- |
| Our str-em vs the per-question ALCE scores of C1 (`critique/score_per_question.py`) | identical on 948 of 948 |
| Coverage at top-1/3/5/20/100 on `asqa_eval_gtr_top100.json` vs ALCE Table 12 (GTR) | 35.1 / 50.7 / 56.8 / 70.3 / 78.4, identical |
| Oracle file, coverage at 1/3/5 vs ALCE Table 12 | 63.8 / 72.8 / 78.4, identical |
| Our matcher vs ALCE's per-passage `answers_found` labels | 15,920 of 15,920 agree |
| C1 rerun from this account (job 55482) vs `asqa_cands_0_end.json` | 200 of 200 answers identical, 2.4 s/question |

### What the pool holds (948 questions)

| Passages offered | All together | Best single passage |
| --- | --- | --- |
| top 1 | 35.1 | 35.1 |
| top 5 | 56.8 | 51.2 |
| top 10 | 64.7 | 55.4 |
| top 20 | 70.3 | 59.1 |
| top 100 | 78.4 | 63.8 |

Self-RAG writes one candidate per passage, so the right-hand column is the limit of a perfect pick.

### The chain on Set A+B (200 questions, C1)

| Stage | Coverage |
| --- | --- |
| The 5 passages together | 59.5 [54.4, 64.4] |
| Best single passage of the 5 | 53.1 [48.5, 57.8] |
| Passage Self-RAG picked | 36.4 [31.9, 40.9] |
| Answer (str-em) | 30.9 [27.0, 35.0] |

The pick is no better than the rank-1 passage (35.7). The answer keeps about 85% of what its passage holds;
this ratio was 0.83 to 0.88 in every run below except the oracle run (0.75).

### Controls (Set A+B, paired with C1)

| Run | str-em | Coverage of passage picked | Best single passage offered | Words | s/question |
| --- | --- | --- | --- | --- | --- |
| C1, GTR top 5 | 30.9 | 36.4 | 53.1 | 29.1 | 2.4 |
| Oracle top 5 (job 55484) | **+9.2 [6.0, 12.6]** | +17.0 [13.3, 20.8] | +13.7 [10.5, 17.1] | -0.2 | 2.5 |
| GTR top 10 (job 55487) | -0.2 [-2.7, 2.3] | -1.6 [-4.3, 1.0] | +4.7 [2.6, 7.1] | +0.5 | 3.5 |
| GTR top 20 (job 55491) | -1.0 [-4.7, 2.6] | -2.6 [-5.9, 0.8] | +9.3 [6.4, 12.5] | +1.0 | 5.5 |

The oracle uses gold answers and is a ceiling, not a method. ALCE reports the same swap for ChatGPT as EM
recall 40.4 to 48.9 (its Table 19).

### Several passages per answer, offline bound (Set A+B, no new generation)

Top-k saved candidates of C1 by Self-RAG's score, joined, duplicate sentences removed
(`retrieval/stitch_candidates.py`).

| Built from | str-em | vs C1 | Words |
| --- | --- | --- | --- |
| top 1 (C1) | 30.9 | | 29 |
| top 2 | 41.0 | +10.1 [7.4, 13.1] | 56 |
| top 3 | 45.3 | +14.5 [11.3, 17.9] | 84 |
| top 5 | 49.8 | +18.9 [15.5, 22.8] | 123 |

str-em is a recall metric and rewards length, so this is a bound with a known confound, not a result. A real
version needs a change to when a new segment starts (`run_long_form_static.py:256`, `:143-176`) and two fixes
to citation numbering (`:497-514`). Not built.

### What Part 1 decided

| Option | Evidence | Decision |
| --- | --- | --- |
| More passages, Self-RAG's own scoring | no gain, 1.5x to 2.3x time | rejected |
| Rerank the pool with an outside model | ceiling +9.2 at no length cost; the generator cannot rank | tested in Part 2 |
| Several passages per answer | +10.1 with doubled length; needs changes inside the beam loop | reported as a bound |

## Part 2. Reranking test (21 Nov 2026)

Plan, metrics and rules: `retrieval/PREREG.md`, fixed before the first reranker job. No deviation occurred.
Decision records written between steps: `retrieval/records/`.

Rerankers: R1 = `cross-encoder/ms-marco-MiniLM-L-12-v2` @ 7b02352, R2 = `BAAI/bge-reranker-v2-m3` @ 953dc6f.
Cost on an L40S for all 100 passages of a question: 0.055 s (R1), 0.170 s (R2) (job 56491).

### Step 1: configuration choice on Set A, from passage coverage only

| Configuration | mean@5 | vs GTR |
| --- | --- | --- |
| GTR top 5 | 24.0 | |
| R1, K=20 | 25.7 | +1.7 [-0.2, 3.6] |
| R1, K=100 | 25.2 | +1.2 [-0.9, 3.3] |
| **R2, K=20** | 28.1 | **+4.1 [2.4, 5.9]** |
| R2, K=100 | 28.5 | +4.4 [2.1, 6.7] |

R2-100 and R2-20 are within 0.5; the tie rule (smaller K) selects R2, K=20.

### Results for R2, K=20 (differences from C1)

| | Set A (dev) | Set B (test) | Set C (confirmation, n=748) |
| --- | --- | --- | --- |
| mean@5 of the passages given | +4.1 [2.4, 5.9] | +4.6 [2.3, 6.8] | **+3.1 [2.4, 3.8]** |
| Best single passage given | +5.1 [1.4, 9.4] | +3.7 [-0.5, 8.0] | +2.7 [1.4, 3.9] |
| All 5 passages together | +5.7 [1.4, 10.4] | +3.4 [-1.9, 8.2] | +3.3 [1.8, 4.9] |
| Passage Self-RAG picked | +4.1 [-0.5, 8.6] | -0.4 [-5.1, 4.3] | +2.5 [0.8, 4.1] |
| **str-em** | +2.5 [-2.5, 7.7] | +0.7 [-3.8, 5.3] | **+1.7 [0.3, 3.1]** |
| Best candidate str-em | +1.0 [-2.5, 4.9] | +1.9 [-0.2, 4.4] | +2.4 [1.3, 3.6] |
| Words | -0.2 | -0.4 | -0.8 [-2.0, 0.4] |

Set A+B: job 56495. Set C: job 56496; C1 str-em 30.02 (the value Division 2 reports for Set C), reranked 31.73.
Self-RAG time is unchanged (2.3 to 2.5 s/question).

### Verdicts by the rules fixed in advance

| Rule | Outcome |
| --- | --- |
| Gate G1 (Set A, mean@5 interval above 0) | passed |
| H1 (Set B, mean@5 higher) | confirmed |
| H2 (Set B, str-em higher) | not supported: interval includes 0 |
| Set C (run once because the Set B point estimate was positive) | confirmed: str-em interval above 0, length unchanged |

Prediction fixed in advance, str-em change = 0.85 x change in coverage of the picked passage:

| Set | Predicted | Observed |
| --- | --- | --- |
| A | +3.5 | +2.5 |
| B | -0.4 | +0.7 |
| A+B | +1.6 | +1.6 |
| C | +2.1 | +1.7 |

Expectations written beforehand: "union@5 may fall" was wrong (it rose on every set); "gain well below the
oracle's +9.2" was right; "rank1 rises most" held on Set A only; K=20 vs K=100 was a tie.

### Exploratory (Set A only, not in the plan)

Picking the candidate written from the reranker's first passage instead of Self-RAG's own pick:
+0.6 [-4.0, 5.4]. No evidence of a gain; not run on Set B or Set C (`retrieval/explore_pick.py`).

## What limits the gain

After reranking, on Set C the passage Self-RAG picks holds 38.5% of gold answers while the best of its five
holds 54.3% and the five together hold 60.5%. The reranker moves all three by about 3 points; the two gaps
between them stay. The first gap is candidate picking (the critique and NLI-grounding divisions); the second
is one passage per answer.

## Not done

- ROUGE-L, MAUVE and citation recall/precision of the reranked runs (T5-11B scorer not available on this
  account). Files ready for scoring: `results/runs/retrieval/asqa_rrR2K20_0_end.json`,
  `asqa_rrR2K20C_0_end.json`.
- A retrieval trigger for multi-passage answers.

## Sources

- Asai et al., Self-RAG, ICLR 2024: https://arxiv.org/abs/2310.11511
- Gao et al., ALCE, EMNLP 2023 (Tables 12 and 19, App. G.1): https://arxiv.org/abs/2305.14627
- Reranker model cards: https://huggingface.co/BAAI/bge-reranker-v2-m3 ,
  https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2
