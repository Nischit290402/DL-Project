# NLI grounding (D2), Step 2: Set A diagnostics G1–G4

Set A only (100 questions, 500 candidates). Script: `grounding/diagnostics.py` (CPU, standard library).
Full numbers: `results/ground_diag_A.json`. Hand-labelling sheet: `results/ground_diag_A_G4.csv`.
Inputs: DeBERTa grounding `results/runs/cand_ground_A.jsonl`, exact ALCE metrics
`results/runs/cand_metrics_AB_exact.jsonl`, T5 verdicts on graded claims.

Grounding aggregations g (DeBERTa, each ALCE sentence vs the candidate's own passage): `min` / `mean` / `first`
of p_ent over sentences; `full` = whole answer as one hypothesis; `soft_mean` = mean of p_ent + 0.5·p_neu;
`nocon_min` = min of 1 − p_con (the "filter" signal of mode d).

**Verdict: go. Both stop rules pass for `mean`; choose `mean` as the aggregation for Step 3.**

## G1: an independent but related checker

Sentence level, 476 graded claims with a T5 verdict (DeBERTa label = p_ent ≥ 0.5):

| | supported rate | agreement with T5 | Cohen's κ with T5 |
| --- | --- | --- | --- |
| DeBERTa | 39.9% | **78.8%** | **0.575** |
| Self-RAG ISSUP (`[Fully supported]`) | 76.9% | 63.7% | 0.275 |

DeBERTa vs ISSUP: agreement 55.5%, κ 0.196. DeBERTa is much closer to T5 than Self-RAG's own grade is, while
being stricter than T5 (39.9% vs 49.8% supported).

Candidate level, Pearson correlation with exact ALCE citation recall (500 candidates): ISSUP `ground_score` 0.394,
released score 0.375, DeBERTa `mean` **0.654**, `full` 0.634, `soft_mean` 0.631, `first` 0.566, `min` 0.551,
`nocon_min` 0.370. (T5 on graded claims: 0.941, the grader itself.)

## G2: does it order candidates by str-em? (stop if the interval includes 0.5)

Within-question pairs with different str-em (365 pairs), 95% bootstrap over questions.

| Signal | Concordance | 95% CI | Fix rate* |
| --- | --- | --- | --- |
| Released score (reference) | **0.682** | [0.609, 0.753] | — |
| ISSUP `ground_score` | 0.677 | [0.603, 0.749] | 0.00 |
| DeBERTa `mean` | 0.614 | [0.535, 0.689] | 0.43 |
| DeBERTa `min` | 0.608 | [0.529, 0.683] | 0.44 |
| DeBERTa `soft_mean` | 0.608 | [0.532, 0.680] | 0.42 |
| DeBERTa `first` | 0.605 | [0.526, 0.683] | 0.41 |
| DeBERTa `full` | 0.578 | [0.499, 0.653] | 0.42 |
| DeBERTa `nocon_min` | 0.560 | [0.503, 0.624] | 0.46 |
| T5 verdict (upper-bound reference) | 0.577 | [0.518, 0.638] | 0.38 |

\* Fix rate: among the 116 pairs the released score orders wrong, the share the signal orders right.

Every DeBERTa aggregation except `full` passes (interval above 0.5), but all are weaker than the released score,
and the fix rate is below 0.5. **Grounding will not repair str-em mistakes.** In a combined score it can only cost
str-em, so keep the added weight small and test str-em as non-inferiority.

## G3: does it order candidates by citation recall? (stop if the interval includes 0.5)

Within-question pairs with different exact citation recall (527 pairs).

| Signal | Concordance | 95% CI | Fix rate* (131 pairs) |
| --- | --- | --- | --- |
| Released score (reference) | 0.751 | [0.698, 0.802] | — |
| ISSUP `ground_score` | 0.757 | [0.707, 0.806] | 0.05 |
| **DeBERTa `mean`** | **0.841** | **[0.795, 0.878]** | **0.79** |
| DeBERTa `soft_mean` | 0.839 | [0.787, 0.882] | 0.80 |
| DeBERTa `full` | 0.808 | [0.762, 0.852] | 0.81 |
| DeBERTa `first` | 0.778 | [0.733, 0.817] | 0.66 |
| DeBERTa `min` | 0.761 | [0.713, 0.807] | 0.78 |
| DeBERTa `nocon_min` | 0.645 | [0.584, 0.705] | 0.66 |
| T5 verdict (upper-bound reference) | 0.893 | [0.862, 0.922] | 0.81 |

DeBERTa `mean` orders citation better than Self-RAG's own score (0.841 vs 0.751, intervals barely overlap). It
repairs **79%** of the pairs the released score gets wrong, nearly as many as the grader itself (81%), and it
closes about 63% of the gap from the released score to T5 ((0.841 − 0.751) / (0.893 − 0.751)).
`first` (the ISSUP-style scope) is clearly weaker than `mean` (0.778): the sentences ISSUP never grades matter.
`nocon_min` is weak: low contradiction is not the same as being supported, so **drop mode (d) filtering**.

## G4: where ISSUP and DeBERTa disagree most (hand-labelling pending)

The 30 candidates with the largest |ISSUP support score − DeBERTa p_ent| on the first sentence are in
`results/ground_diag_A_G4.csv` (question, sentence, passage, both scores, T5 verdict, empty columns
`human_supported`, `error_type`, `note`).

- 29 / 30 are "ISSUP says fully supported, DeBERTa says not entailed"; only 1 is the reverse.
- The T5 verdict sides with DeBERTa on 14 and with ISSUP on 16. **Neither checker is clearly right on its worst
  cases; human labels are needed.**
- Preliminary read of the top 12 (not a substitute for labelling):
  - **DeBERTa misses** support that needs bridging across a passage sentence or a paraphrase: Gary Oldman as
    Sirius Black, Spain winning in 2010, Williamsburg as capital until 1780. It also gives two odd high-contradiction
    calls on near-verbatim sentences (Count of Monte Cristo, p_con 0.99).
  - **ISSUP misses** unsupported specifics: exact dates that are not in the passage (1933 / 1971 gold standard, the
    "Lean on Me" release day, the Phillies date), a wrong "first foreigner" claim, and an invented character name.
  - This is the "extra detail → neutral" risk from plan §6.4, and the 0.99 contradictions are a new failure mode.
    `soft_mean` (half credit for neutral) is the natural hedge if labelling confirms many DeBERTa false negatives.

## Decision for Step 3

1. Aggregation: **`mean`**, chosen on G2 + G3 before looking at any re-ranked metric. `soft_mean` is the only
   alternative worth keeping (ties `mean` on G3), as a single robustness row.
2. Modes: (b) add `released + μ·g_mean` and (c) mix `support := (1−α)·ISSUP + α·g_mean`. Drop (a) replace (it loses
   ISSUP's str-em ordering, see G2) and (d) filter (weak signal, see G3).
3. Grid: μ ∈ {0.1, 0.25, 0.5, 0.75, 1.0}, α ∈ {0.25, 0.5, 0.75, 1.0}. Pick from the middle of a plateau, commit
   the setting before touching Set B.
4. Hypothesis for Set B / C: citation recall up (primary); str-em not worse than a pre-set margin (secondary,
   non-inferiority), since G2 says grounding cannot raise str-em.
5. Stop rules for G2/G3 as written in the plan both pass. Note that G2 is better treated as a str-em risk check
   than as a stop rule: D2 targets citation.
