# Critique division: Self-RAG ranking as the paper intended (no tuning)

**Summary.** The released `run_long_form_static.py` ranks candidates by relevance + support only: the utility term
is always 0 and the sequence score is never passed to the scorer. Restoring both, with the paper/README weights
(1.0 / 1.0 / 0.5, `--use_seqscore`) and no tuning, was tested on all 948 ASQA questions under a test plan written
before the run. It changes 272 answers and **significantly improves citation recall (+2.85 [+1.05, +4.67]), citation
precision (+2.97 [+1.24, +4.79]) and ROUGE-Lsum (+0.58 [+0.11, +1.09])** over C1. str-em rises by +0.64 [-0.46,
+1.82] (not significant), MAUVE from 75.57 to 79.42, and answers are 4.5 words longer. It is above the paper's
reported 7B numbers on all five ASQA metrics (point estimates; the paper gives no intervals). The 200-question
check that motivated the full run is kept below.

## Why this check (grounds)

- The paper's critique-guided decoding scores each segment with the generation probability plus weighted
  `[IsRel]`, `[IsSup]` and `[IsUse]` terms; the README's ASQA command uses `--use_grounding --use_utility
  --use_seqscore` with weights 1.0 / 1.0 / 0.5.
- In the released code (also upstream), two of those terms never reach the ranking:
  1. Utility: `run_step_generation_batch` reads the utility probabilities by looping over the support tokens
     (`grd_tokens`) instead of `ut_tokens`; the resulting dict has 3 entries, the 5-entry check fails, and
     `utility_score` is always 0.0.
  2. Sequence score: `call_model_beam_batch` calls `run_step_generation_batch` without `use_seqscore`, so it stays
     False and `exp(seq_score)` is never added.
- The question is therefore: does Self-RAG pick better answers when it uses its full critique, as published?
- No weights are searched: they are the paper's. So the evaluation may use all 200 questions of Set A and Set B.

## Method

- Candidates: `results/runs/asqa_cands_0_end.json` (Set A + B, 5 first-step candidates per question, saved with
  `--save_candidates`; the C1 choice is reproduced 200/200).
- Per-candidate ALCE scores: Person 2's `results/runs/critique_p2/candidate_*.json.perq.jsonl` (Colab, ALCE
  `246c476`, `google/t5_xxl_true_nli_mixture`). The baseline row uses the same scores for C1's own choices, so both
  sides come from one environment (C1's Set B citation recall is 69.35 here vs 70.35 in the cluster scoring: one
  question scored differently on Colab, as Person 2 reported).
- Utility as the code intends it: sum(w_i p_i) / sum(p_i), w = (-1, -0.5, 0, 0.5, 1) for Utility:1..5, from
  `utility_probs` (read from the utility tokens). Sequence term: exp(mean token log-probability).
- Re-ranking and lookup: `python3 critique/paper_scoring_check.py results/runs/critique_paper_scoring`;
  intervals: `critique/bootstrap.py` (1000 paired resamples, `--split`).

| Configuration | w_rel | w_sup | w_use | seq score | Picks changed vs C1 (of 200) |
| --- | --- | --- | --- | --- | --- |
| Released (= C1) | 1 | 1 | 0 | off | 0 |
| + utility only | 1 | 1 | 0.5 | off | 7 |
| + sequence score only | 1 | 1 | 0 | on | 65 |
| **Paper-intended (both)** | 1 | 1 | 0.5 | on | **66** |

## Results: paper-intended vs C1 (difference [95% CI], `*` = excludes 0)

| Metric | Set A (100) | Set B (100) | **A + B (200)** |
| --- | --- | --- | --- |
| str-em | -0.23 [-4.07, +3.80] | +3.27 [-1.15, +7.68] | **+1.52 [-1.48, +4.62]** |
| ROUGE-Lsum | +1.42 [+0.09, +3.20] * | +0.37 [-1.75, +2.40] | **+0.89 [-0.28, +2.09]** |
| Citation recall | +4.08 [-1.58, +10.07] | +3.22 [-2.70, +9.32] | **+3.65 [-0.23, +8.07]** |
| Citation precision | +4.25 [-1.25, +10.45] | +3.05 [-2.70, +9.10] | **+3.65 [-0.16, +7.78]** |
| Length (words) | +3.79 [+0.57, +7.07] * | +8.43 [+4.60, +12.36] * | **+6.11 [+3.66, +8.64] *** |

Absolute values on A + B: C1 str-em 30.85, citation recall 68.28, precision 68.94, length 27.6;
paper-intended 32.37, 71.93, 72.59, 33.7.

## Ablation: which restored term matters (A + B, 200 questions)

| Metric | + utility only | + sequence score only | Both |
| --- | --- | --- | --- |
| str-em | +0.25 [-0.83, +1.58] | +1.02 [-1.87, +3.94] | +1.52 [-1.48, +4.62] |
| ROUGE-Lsum | -0.23 [-0.84, +0.32] | +0.81 [-0.36, +2.04] | +0.89 [-0.28, +2.09] |
| Citation recall | -0.25 [-2.00, +1.50] | +3.40 [-0.46, +7.82] | +3.65 [-0.23, +8.07] |
| Citation precision | -0.25 [-2.00, +1.50] | +3.40 [-0.47, +7.51] | +3.65 [-0.16, +7.78] |
| Length | -0.14 [-0.39, +0.13] | +6.03 [+3.56, +8.59] * | +6.11 [+3.66, +8.64] * |

Utility alone changes 7 choices and nothing measurable; the sequence score drives the effect.

## Interpretation

- Unlike the learned weights (Person 2: str-em +2.68, citations about -6.6 on Set B), the paper's own scoring moves
  coverage and citations in the same direction: all four quality point estimates are positive on A + B.
- None is significant on 200 questions. Citation recall and precision are closest (lower bounds -0.23 and -0.16).
- Answers are 22% longer. Longer answers can mention more short answers (str-em) without being better, so any
  claim must report length next to quality.
- What it does not show on its own: that the gain is real. That needed all 948 questions (next section).

## Confirmation on all 948 questions

Test plan, written before the run: the configuration is fixed by the paper, not tuned, so it is evaluated on the
full set without a held-out split. Weights 1.0 / 1.0 / 0.5 with the sequence score; primary metrics citation recall
and str-em; paired comparison against C1 with 95% bootstrap intervals; length and MAUVE reported alongside; the
result is reported whatever it is.

Procedure (NTU TC2, NVIDIA L40S):
1. `--keep_special_tokens --save_candidates` on all 948 questions (`cluster/run_asqa.sbatch`, TAG `full948`).
   Its chosen answers equal C1's on 948/948.
2. Offline re-ranking with `critique/learn_weights.py apply` and the two fixed configs in
   `results/critique_paper_scoring_configs/` (`released.json`, `paper_intended.json`). The released config
   reproduces C1's text on 948/948; the one two-segment answer is kept as is by both.
3. Both re-ranked files scored on the cluster with `cluster/run_critique_score.sbatch` (`--mauve`). The released
   file's scores equal the official C1 scores exactly (str-em 30.19, citation 66.53 / 67.47, MAUVE 75.57).

Results (948 questions, paper-intended minus C1, 95% paired bootstrap, `*` = interval excludes 0):

| Metric | C1 | Paper-intended scoring | Difference [95% CI] |
| --- | --- | --- | --- |
| str-em | 30.19 | 30.84 | +0.64 [-0.46, +1.82] |
| ROUGE-Lsum | 35.79 | 36.37 | **+0.58 [+0.11, +1.09]** * |
| Citation recall | 66.53 | 69.39 | **+2.85 [+1.05, +4.67]** * |
| Citation precision | 67.47 | 70.44 | **+2.97 [+1.24, +4.79]** * |
| MAUVE | 75.57 | 79.42 | - (whole-set metric) |
| Length (words) | 29.96 | 34.45 | +4.49 [+3.44, +5.48] * |
| Answers changed | - | 272 of 948 | - |

Against the paper (Table 2, Self-RAG 7B; the author's own ALCE output gives citation recall 66.97 and precision
67.82, see `results/asqa_reproduction.md` on the swapped Table 2 columns):

| Metric | Paper | Our C1 | Paper-intended scoring |
| --- | --- | --- | --- |
| str-em | 30.0 | 30.19 | 30.84 |
| ROUGE-Lsum | 35.7 | 35.75 | 36.37 |
| MAUVE | 74.3 | 75.57 | 79.42 |
| Citation recall | 66.97 | 66.64 | 69.39 |
| Citation precision | 67.82 | 67.57 | 70.44 |

Our C1 column uses ALCE's own `eval.py`; the last column uses the per-question scorer, which matches ALCE except
for one borderline citation judgement out of 948 and a small ROUGE difference (`results/critique_p1.md`).

Reading:
- Of the two primary metrics, citation recall improves significantly; str-em does not. Citation precision and
  ROUGE-Lsum also improve significantly.
- The gain cannot come from length alone: citation recall is the share of sentences supported by their cited
  passage, and longer answers do not make that share easier to raise. str-em, which longer answers can inflate, is
  the metric that does not change significantly.
- The Set A + B estimates (200 questions) were in the same direction and larger (+3.65 citation recall); with 948
  questions the intervals narrow enough to exclude 0.
- Cost: no retraining and no extra model. The sequence score is already computed by vLLM, and re-ranking takes
  about 0.13 ms per question.

Files: `results/runs/critique_paper_scoring/full948_{released,paper}.json.{perq.jsonl,summary.json}`.
The 13 MB candidate file (`out/asqa_full948_0_end.json`) and the re-ranked answer files are on the cluster and
Person 1's laptop, not committed.
