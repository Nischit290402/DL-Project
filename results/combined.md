# Combined system: final results

**Summary.** The final system stacks the three components that each improved on C1 in their own held-out test:
cross-encoder reranked passages (retrieval division), the paper's full critique score (critique division) and the
D2-v2 NLI grounding term with its relevance gate (grounding division). Nothing was re-tuned for the combination.
Under the test plan written before the combined run (`combined/PREREG.md`), on the 748 held-out questions of Set C
it raises **str-em by +2.84 [+1.10, +4.65]** and **citation recall by +11.44 [+8.47, +14.41]** over C1; both
pre-registered hypotheses are confirmed. Citation precision (+11.34) and ROUGE-Lsum (+1.00) also improve
significantly, and length does not change significantly (+1.25 [-0.20, +2.78]). Each component adds a significant
gain on top of the previous ones. On all 948 questions the system is above the paper's reported Self-RAG 7B numbers
on all five ASQA metrics.

## Setup

- Self-RAG 7B, upstream README long-form settings with `--keep_special_tokens` (C1), NVIDIA L40S, NTU TC2.
- Passages: the retrieval division's Self-RAG runs on the cross-encoder reranked GTR top 20 (top 5 kept),
  `results/runs/retrieval/asqa_rrR2K20_0_end.json` (Set A + B) and `asqa_rrR2K20C_0_end.json` (Set C).
- Ranking: `combined/rerank_combined.py`, which imports each division's code unchanged:
  score = relevance + support + 0.5 utility + exp(sequence score) + 0.75 (g - support), among candidates that pass
  the bge-reranker-base relevance gate (delta 1). g = mean DeBERTa entailment of the candidate's sentences.
- Scoring: exact ALCE metrics of every candidate (`cluster/run_combined_prep.sbatch`, job 56550), then lookup.
  The final 948 answers were also scored directly (`cluster/run_critique_score.sbatch --mauve`): identical to the
  lookup on every metric (str-em 33.69, ROUGE-Lsum 36.82, citation 77.49 / 78.29, length 31.25).
- GPU checks: all 1000 Set A + B and 3735 of 3740 Set C candidates have exact citation scores (the other 5 are the
  candidates of the one multi-segment question, which every system keeps as released); 0 str-em / ROUGE
  mismatches; the retrieval runs' own choices reproduce their direct scores on 200/200 and 747/747 questions.
- Offline checks: `--critique released` with no other component changes 0 answers on both sets; the grounding
  term alone reproduces the grounding division's committed D2-v2 choices on 748/748 GTR questions.
- Intervals: paired bootstrap over questions, 1,000 resamples, seed 0 (`critique/bootstrap.py`). `*` = the interval
  of the difference excludes 0.

## Primary result: Set C (748 questions)

| Metric | C1 | Final system | Difference [95% CI] |
| --- | --- | --- | --- |
| str-em (H1) | 30.02 | **32.86** | **+2.84 [+1.10, +4.65]** * |
| Citation recall (H2) | 65.93 | **77.38** | **+11.44 [+8.47, +14.41]** * |
| Citation precision | 66.94 | 78.28 | +11.34 [+8.57, +14.52] * |
| ROUGE-Lsum | 35.89 | 36.88 | +1.00 [+0.10, +1.83] * |
| Length (words) | 30.58 | 31.84 | +1.25 [-0.20, +2.78] |

H1 and H2 confirmed.

## Ablation ladder (Set C, each row against the row above)

| Step | str-em | ROUGE-Lsum | Citation recall | Citation precision | Length |
| --- | --- | --- | --- | --- | --- |
| R1 + retrieval (vs C1) | **+1.71 [+0.35, +3.18]** * | -0.33 [-1.09, +0.41] | **+3.15 [+0.54, +5.89]** * | **+3.22 [+0.63, +6.29]** * | -0.82 [-2.01, +0.34] |
| R2 + critique (vs R1) | +0.47 [-0.75, +1.75] | **+0.81 [+0.22, +1.43]** * | **+3.91 [+1.81, +5.98]** * | **+4.24 [+2.17, +6.11]** * | +4.29 [+3.17, +5.49] * |
| R3 + grounding (vs R2) | +0.67 [-0.63, +2.07] | +0.51 [-0.23, +1.25] | **+4.38 [+2.18, +6.48]** * | **+3.88 [+1.80, +6.02]** * | -2.22 [-3.56, -1.05] * |

Absolute values: C1 30.02 / 35.89 / 65.93 / 66.94; R1 31.73 / 35.56 / 69.09 / 70.16; R2 32.19 / 36.37 / 73.00 /
74.40; R3 32.86 / 36.88 / 77.38 / 78.28 (str-em / ROUGE-Lsum / citation recall / precision).

Off the ladder: retrieval + grounding without the critique score gives str-em +1.66 [-0.09, +3.41] and citation
recall +10.03 [+7.03, +13.07] vs C1, below the full system on every quality metric.

Reading: retrieval is what moves correctness (better passages contain more gold answers); the critique and
grounding terms mostly move citation quality, by choosing candidates that their own passage supports. The critique
score lengthens answers (+4.3 words) and the grounding term shortens them again (-2.2), so the final answers are
about the same length as C1's.

## All 948 questions (secondary)

Set A + B were used by the divisions to fix their settings, so these numbers include tuning questions.

| Metric | C1 | Final system | Difference [95% CI] |
| --- | --- | --- | --- |
| str-em | 30.19 | 33.69 | +3.50 [+1.97, +5.05] * |
| ROUGE-Lsum | 35.79 | 36.82 | +1.03 [+0.29, +1.82] * |
| Citation recall | 66.53 | 77.49 | +10.95 [+8.29, +13.49] * |
| Citation precision | 67.47 | 78.29 | +10.82 [+7.96, +13.42] * |
| MAUVE | 75.57 | 76.16 | - (whole-set metric) |
| Length (words) | 29.96 | 31.25 | +1.29 [+0.04, +2.61] * |

Ladder on 948: + retrieval str-em +1.69 [+0.37, +2.95]; + critique citation recall +4.18 [+2.39, +6.04];
+ grounding citation recall +4.56 [+2.70, +6.49].

## Against the paper (Table 2, Self-RAG 7B)

| Metric | Paper | Our C1 | Final system |
| --- | --- | --- | --- |
| str-em | 30.0 | 30.19 | **33.69** |
| ROUGE-Lsum | 35.7 | 35.75 | **36.82** |
| MAUVE | 74.3 | 75.57 | **76.16** |
| Citation recall | 66.97 | 66.64 | **77.49** |
| Citation precision | 67.82 | 67.57 | **78.29** |

Paper citation values are from the author's own ALCE output, as the two citation columns of Table 2 appear
swapped (`results/asqa_reproduction.md`). The paper reports no intervals.

## Cost

No retraining. Added at inference: one cross-encoder pass over the GTR top 20 (retrieval division: 0.17 s per
question), one DeBERTa NLI pass over the 5 candidates (grounding division: 0.09 s per question), one
bge-reranker-base pass over the 5 candidate passages (not timed separately), and a CPU re-ranking step under 1 ms
per question. Self-RAG generation stays at
about 2.5 s per question on an L40S.

## Files

| File | Content |
| --- | --- |
| `combined/PREREG.md` | Test plan, written before the combined run |
| `combined/rerank_combined.py`, `combined/run_ladder.sh`, `cluster/run_combined_prep.sbatch` | Code; steps in `combined/README.md` |
| `results/runs/combined/cand_*_rr_*.jsonl` | Per-candidate exact metrics, grounding and relevance on the reranked passages |
| `results/runs/combined/ladder/` | Per-question scores, bootstrap tables and configs of every ladder row |
| `results/runs/combined/final_948.json` (+ `.summary.json`) | Final system's answers for all 948 questions and their direct ALCE scores |
