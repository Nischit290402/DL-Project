# Critique division, Person 1: baselines and honesty of Self-RAG's self-grades

**Summary.** C1 (`--keep_special_tokens`) beats C0 on citations only: recall +16.3 [+14.6, +18.1], precision
+3.5 [+2.3, +4.7], with str-em, ROUGE-L and length unchanged. Self-RAG's own support grade is overconfident: it
grades 94% of its answers `[Fully supported]`, but an independent NLI checker agrees only 75% of the time on
chosen answers and 57% across all candidates (calibration error 0.255). Temperature scaling (Fix 1) halves the
calibration error on unseen questions but lowers str-em by 1.75 [-3.75, -0.25], so it is not adopted. Averaging
or taking the lowest of several grades (Fix 2) is not applicable: the model writes at most one grade per segment.

Code, run guide and commands: `critique/README.md` (branch `critique-p1`). All runs on NTU TC2, NVIDIA L40S.

## Setup

- Model and settings: Self-RAG 7B, upstream README long-form settings (`--mode always_retrieve --ndocs 5
  --threshold 0.2 --use_grounding --use_utility --use_seqscore --max_depth 7 --beam_width 2`), as in
  `cluster/run_asqa.sbatch`.
- Baselines: C0 = released code (`results/runs/asqa_baseline_0_end.json`), C1 = C0 + `--keep_special_tokens`
  (`results/runs/asqa_keeptok_0_end.json`), 948 ASQA questions each.
- Test sets: Set A and Set B, 100 questions each, no overlap, seed 6103 (`critique/splits/`).
- Scoring: ALCE metric code at commit `246c476`, applied one question at a time
  (`critique/score_per_question.py`). Citation metrics use `google/t5_xxl_true_nli_mixture`.
- Intervals: 95% paired bootstrap over questions, 1000 resamples (`critique/bootstrap.py`). `*` = the interval of
  the difference excludes 0. Decoding is greedy, so the interval reflects only which questions were sampled.

### Scorer check against ALCE's own numbers (948 questions)

| Run | str-em | ROUGE-Lsum | Cit. rec | Cit. prec | MAUVE | Length |
| --- | --- | --- | --- | --- | --- | --- |
| C1, ALCE `.score` | 30.19 | 35.75 | 66.64 | 67.57 | 75.57 | 29.96 |
| C1, per-question scorer | 30.19 | 35.79 | 66.53 | 67.47 | 75.57 | 29.96 |
| C0, ALCE `.score` | 30.13 | 35.83 | 50.22 | 63.92 | 76.09 | 29.99 |
| C0, per-question scorer | 30.13 | 35.79 | 50.22 | 63.92 | 76.09 | 29.99 |

C0 matches exactly. C1 citation scores are 0.105 lower, which is 100/948: one answer of 948 on which the T5
checker flips between runs. ROUGE-Lsum differs by under 0.05 because ALCE reports the median of its own bootstrap,
while we average per question. MAUVE is a whole-set metric, so it has no per-question value or interval.

## Result 1. Baselines: C0 vs C1 (948 questions)

| Metric | C0 | C1 | C1 - C0 [95% CI] |
| --- | --- | --- | --- |
| str-em | 30.13 [28.12, 32.12] | 30.19 [28.26, 32.26] | +0.06 [-0.21, +0.39] |
| ROUGE-Lsum | 35.79 [34.87, 36.77] | 35.79 [34.85, 36.75] | -0.01 [-0.09, +0.09] |
| Citation recall | 50.22 [47.51, 53.09] | 66.53 [63.69, 69.05] | **+16.31 [+14.57, +18.08]** * |
| Citation precision | 63.92 [60.86, 66.88] | 67.47 [64.55, 70.22] | **+3.54 [+2.34, +4.72]** * |
| Length (words) | 29.99 [28.60, 31.32] | 29.96 [28.60, 31.29] | -0.03 [-0.17, +0.10] |
| MAUVE | 76.09 | 75.57 | - |
| Seconds / question | 3.37 (A40) | 2.54 (L40S) | - |

C1 on the test sets, for reference (every comparison below is paired on the same questions):

| Set | str-em | ROUGE-Lsum | Cit. rec | Cit. prec | Length |
| --- | --- | --- | --- | --- | --- |
| A | 27.78 [22.55, 32.97] | 33.20 [30.62, 35.97] | 67.22 [58.22, 75.25] | 67.22 [58.33, 75.33] | 28.04 |
| B | 33.92 [28.32, 39.73] | 37.62 [34.75, 40.75] | 70.35 [61.67, 78.50] | 71.67 [63.17, 79.67] | 27.21 |

Per-question C0 and C1 scores, for other divisions' intervals:
`results/runs/asqa_baseline_0_end.json.perq.jsonl`, `results/runs/asqa_keeptok_0_end.json.perq.jsonl`
(one line per question, keyed by `q_idx` = position in `asqa_eval_gtr_top100.json`).

## Result 2. Honesty check: does an outside checker agree with Self-RAG's support grade?

Method (`critique/nli_verdicts.py`, `critique/honesty_check.py`): for every `[IsSup]` grade the model wrote, the
text it graded (everything before the grade) is checked against the passage the candidate was written from, with
the same T5-11B NLI model and prompt that ALCE uses for citation recall. Verdict 1 = supported.

### 2a. Chosen answers, C1, 948 questions (labels only)

| Model's grade | Answers | Checker says supported |
| --- | --- | --- |
| `[Fully supported]` | 890 | 74.8% |
| `[Partially supported]` | 15 | 26.7% |
| `[No support / Contradictory]` | 42 | 26.2% |

One question (two segments) is skipped by design. In 336 of 948 C1 answers the model keeps writing after its
single grade (`[Continue to Use Evidence]`), so those later sentences are never graded.

### 2b. All 5 candidates, Set A + B (1000 candidates, 975 with a grade)

Candidates saved with `--save_candidates` (`run_long_form_static.py`, recording only). The 200 chosen answers are
identical to C1's, and re-picking them from the saved scores reproduces all 200.

| Model's grade | All candidates: n | Checker agrees | Chosen only: n | Checker agrees |
| --- | --- | --- | --- | --- |
| `[Fully supported]` | 733 | 57.0% | 191 | 76.4% |
| `[Partially supported]` | 57 | 21.1% | 5 | 40.0% |
| `[No support / Contradictory]` | 185 | 23.2% | 4 | 25.0% |

The grade separates supported from unsupported text (57% vs about 22%), but it is too generous, and most of all on
the candidates it is used to rank against each other.

### 2c. Calibration on Set B (489 grades)

Support score s = P(Fully)/sum + 0.5 x P(Partially)/sum, the formula `run_long_form_static.py` ranks with.

| Model score bin | n | Mean model score | Checker agrees |
| --- | --- | --- | --- |
| 0.0-0.1 | 17 | 0.043 | 0.059 |
| 0.1-0.2 | 21 | 0.148 | 0.143 |
| 0.2-0.3 | 25 | 0.264 | 0.240 |
| 0.3-0.4 | 19 | 0.355 | 0.421 |
| 0.4-0.5 | 21 | 0.453 | 0.238 |
| 0.5-0.6 | 39 | 0.548 | 0.385 |
| 0.6-0.7 | 37 | 0.659 | 0.270 |
| 0.7-0.8 | 41 | 0.750 | 0.341 |
| 0.8-0.9 | 72 | 0.856 | 0.389 |
| 0.9-1.0 | 197 | 0.963 | 0.726 |

ECE 0.2548, Brier 0.2853. Low scores are roughly honest; from 0.4 upward the model is overconfident by 16 to 47
points.

## Result 3. Fix 1: temperature scaling of the support grade (negative)

Grade probabilities p_i -> p_i^(1/T), renormalised. T chosen on Set A by lowest Brier score (grid 0.25 to 10):
**T = 4.0**. Judged on Set B only.

Calibration, Set B:

| | ECE | Brier |
| --- | --- | --- |
| Released (T = 1) | 0.2548 | 0.2853 |
| Fix 1 (T = 4) | **0.1210** | **0.2296** |

Answer quality: answers re-picked from the saved candidates with the scaled grade (`critique/rerank.py`); every
other part of the ranking unchanged. Fix 1 changes 13 of 200 answers.

| Metric, Set B (test) | Released | Fix 1 | Fix 1 - released [95% CI] |
| --- | --- | --- | --- |
| str-em | 33.92 | 32.17 | **-1.75 [-3.75, -0.25]** * |
| ROUGE-Lsum | 37.62 | 37.11 | -0.52 [-1.92, +0.89] |
| Citation recall | 70.35 | 69.92 | -0.43 [-3.53, +2.83] |
| Citation precision | 71.67 | 70.92 | -0.75 [-4.25, +3.00] |
| Length | 27.21 | 26.39 | -0.82 [-3.06, +0.86] |

| Metric, Set A (tune) | Released | Fix 1 | Fix 1 - released [95% CI] |
| --- | --- | --- | --- |
| str-em | 27.78 | 28.28 | +0.50 [+0.00, +1.50] |
| ROUGE-Lsum | 33.20 | 33.44 | +0.24 [-0.22, +0.85] |
| Citation recall | 67.22 | 67.22 | +0.00 [-3.00, +3.00] |
| Citation precision | 67.22 | 67.22 | +0.00 [-3.00, +3.00] |
| Length | 28.04 | 28.69 | +0.65 [+0.06, +1.46] * |

Conclusion: Fix 1 makes the grade more honest but picks worse answers on unseen questions, so it does not beat C1
and is not run on the full set. Ranking needs the right order between candidates, not calibrated values;
T = 4 squeezes all support scores into 0.3-0.8, which shrinks the support term next to relevance.

## Result 4. Fix 2: average or lowest of several grades (not applicable)

The ranking reads only the first `[IsSup]` grade of each candidate; Fix 2 would combine all of them. In the saved
candidates, 975 of 1000 have exactly one grade and 25 have none; none has two or more. Re-picking with the average
changes 0 of 200 answers. Fix 2 therefore cannot change any choice, and the GPU scoring runs were skipped.

## Files

| File | Content |
| --- | --- |
| `results/runs/asqa_baseline_0_end.json.perq.jsonl` | C0 scores per question (948) |
| `results/runs/asqa_keeptok_0_end.json.perq.jsonl` | C1 scores per question (948) |
| `results/runs/asqa_keeptok_0_end.json.verdicts.jsonl` | Result 2a: checker verdict per C1 grade |
| `results/critique_honesty.json` | Results 2b, 2c, 3: agreement, calibration tables, T |
| `out/asqa_cands_0_end.json` (cluster, not committed, 2.9 MB) | All candidates of Set A + B with grade probabilities |
| `out/rr_released.json`, `out/rr_fix1.json` (+ `.perq.jsonl`) (not committed) | Re-picked answers for Result 3 |
