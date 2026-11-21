# Critique division, Person 2: learned reranking weights

## Setup

We reuse the five first-step candidates that Self-RAG 7B generated for each of the
200 ASQA questions in the fixed A/B splits. The reference is `rr_released`,
the C1 output on those questions. With the effective C1 settings
(`w_rel=1`, `w_sup=1`, `w_use=0`, sequence score off), the offline
selector reproduces the selected passage, formatted answer and cited evidence on 200/200 questions.

For each passage position we export one ALCE answer file, keeping that candidate's
own cited evidence. We evaluate every candidate once, then look up scores by
question ID and passage position. The oracle chooses the highest-scoring
candidate for each question and metric using the reference answers. It measures
the ceiling of this fixed candidate pool.

The weight grid tests relevance in `{0, 0.5, 1, 2}`, support in
`{0, 0.5, 1, 2, 3}`, utility in `{0, 0.5, 1}`, and the sequence
term off/on. We search with raw supports and separately with Person 1's
fixed temperature of 4. Selection uses Set A str-em. The full procedure
breaks ties using citation recall, citation precision, ROUGE-Lsum, then
distance to C1. After selection, we report differences from C1 with
paired 95% bootstrap intervals on Set B.

Commands are in [the run guide](../critique/README.md#person-2-select-and-validate-reranking-weights).

## Candidate scores and oracle

We collected full ALCE scores for 1,000 candidates with no missing citation
values. Colab provided the GPU for candidate evaluation. We used ALCE commit
`246c476` and `google/t5_xxl_true_nli_mixture` in BF16 with a device map
that permits CPU and disk offload.

| Metric | A C1 | A oracle | B C1 | B oracle |
| --- | ---: | ---: | ---: | ---: |
| str-em | 27.78 | 40.18 | 33.92 | 49.18 |
| ROUGE-Lsum | 33.20 | 40.77 | 37.62 | 47.01 |
| Citation recall | 67.22 | 87.92 | 70.35 | 85.67 |
| Citation precision | 67.22 | 87.92 | 71.67 | 86.50 |

Each oracle picks candidates separately for its metric using evaluation labels.
On Set B the pool has 15.27 points of str-em headroom.

## Set A selection and Set B results

Both grids selected `w_rel=2`, `w_sup=0`, `w_use=0.5`, with the sequence
term off. On A this rule changed 41 passage choices and obtained str-em
30.53, ROUGE-Lsum 34.27, and citation recall and precision 57.38 each.
On B it changed 39 passage choices and 36 formatted answers.

| Configuration | Support temperature | Set B str-em | Difference vs. C1 [95% CI] |
| --- | ---: | ---: | ---: |
| [Learned weights](critique_p2_configs/learned_weights.json) | 1 | 36.60 | +2.68 [-1.15, +6.88] |
| [Learned weights with calibrated support](critique_p2_configs/learned_weights_calibrated.json) | 4 | 36.60 | +2.68 [-1.15, +6.88] |

The support weight is zero in both configurations, so the two rules select
the same candidates. The table below applies to both rows above. Intervals
use 1,000 paired bootstrap resamples.

| Set B metric | C1 | Learned weights | Difference [95% CI] |
| --- | ---: | ---: | ---: |
| str-em | 33.92 | 36.60 | +2.68 [-1.15, +6.88] |
| ROUGE-Lsum | 37.62 | 39.15 | +1.52 [-0.79, +3.83] |
| Citation recall | 70.35 | 63.77 | -6.58 [-13.58, +0.33] |
| Citation precision | 71.67 | 64.58 | -7.08 [-13.75, +0.08] |
| Length (words) | 27.21 | 29.71 | +2.50 [-1.28, +6.37] |

Offline selection and answer formatting took about 0.06 ms per question
on the local CPU. This measurement covers the reranking step.

## Interpretation

The coverage point estimates increased and the citation point estimates
decreased. Every paired interval includes zero. A cross-environment check
found that rescoring C1's selected candidates with Colab changed the citation
scores of one Set B question. Using those Colab scores for both systems,
the differences were -5.58 for recall and -6.08 for precision, with both
intervals still including zero. The main table uses the handoff's
`rr_released` reference.

The primary str-em interval includes zero, so the conditional 948-question
run was skipped. The oracle shows room to improve coverage with the
existing candidate pool. The selected weights did not establish a reliable
coverage gain and had lower citation point estimates. Because both
searches selected zero support weight, their equality provides no evidence
about the effect of calibration when the support term contributes to ranking.

## Files

| File | Content |
| --- | --- |
| `results/runs/critique_p2/candidate_*.json.perq.jsonl` | Full ALCE scores for the 1,000 candidates, grouped by passage position |
| `results/runs/critique_p2/rr_learned_weights.json` | Reranked answers for the 200 Set A and B questions. Both configs make the same selections |
| `results/runs/critique_p2/rr_learned_weights.set_B.perq.jsonl` | Scores of the 100 held-out selections for paired comparison with C1 |
| `results/critique_p2_configs/` | Both Set A configurations and the Set A oracle |
