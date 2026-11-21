# Combined system: test plan (written before the combined GPU run)

## What is combined, and why each part is in

Only components that improved on C1 in their own held-out test are combined. All their settings were fixed on
Set A + B by the divisions; nothing is tuned for the combination.

| Stage | Component | Fixed setting | Own held-out evidence |
| --- | --- | --- | --- |
| Passages | Retrieval: cross-encoder reranks the GTR top 20, Self-RAG reads the top 5 | `R2`, K = 20 (`retrieval/PREREG.md`) | Set C str-em +1.71 [+0.31, +3.14] (`results/retrieval.md`) |
| Ranking score | Critique: the paper's full critique score (relevance + support + 0.5 utility + exp(sequence score)) | paper / README weights, no tuning | 948 questions: citation recall +2.85 [+1.05, +4.67] (`results/critique_paper_scoring.md`) |
| Ranking score | Grounding: D2 term w_sup * 0.75 * (g - support), g = mean DeBERTa entailment of the candidate's sentences | `mix`, `mean`, alpha 0.75 (`grounding/configs/d2v2_final.json`) | Set C (with the gate): str-em +1.13 [+0.01, +2.28], citation recall +4.39 [+2.31, +6.34] |
| Eligibility | Grounding: D2-v2 relevance gate, bge-reranker-base, delta 1 | as above | as above |

Left out: the self-contradiction penalty (Set C: str-em +0.17 [-0.84, +1.13], citation recall -2.24
[-4.17, +0.01]; no gain). Its confirmed result (disagreement among candidates predicts errors) is an analysis
finding, not a ranking component. Fix 1 (temperature scaling) and Person 2's learned weights are left out: neither
improved on Set B.

How the parts fit: the ranking terms are additive on Self-RAG's own candidate score, so the critique score replaces
the released score and the grounding term is added on top, exactly as each division defined it. The code is
`combined/rerank_combined.py`, which imports each division's functions unchanged. Retrieval changes the passages,
so the combined ranking runs on the retrieval division's committed Self-RAG runs on the reranked passages
(`results/runs/retrieval/asqa_rrR2K20_0_end.json` for Set A + B, `asqa_rrR2K20C_0_end.json` for Set C;
C1 settings, `--save_candidates`).

## Ablation ladder (each row adds one component)

| Row | Passages | Ranking |
| --- | --- | --- |
| R0 = C1 | GTR top 5 | released |
| R1 | reranked top 5 | released |
| R2 | reranked top 5 | paper critique score |
| **R3 = final** | reranked top 5 | paper critique score + D2 grounding + D2-v2 gate |

## Tests

Primary, Set C (748 questions, not used to fix any setting):
- **H1:** R3 vs C1, str-em: the paired 95% interval of the difference lies above 0.
- **H2:** R3 vs C1, citation recall: the paired 95% interval lies above 0.
Also reported, not tests: ROUGE-Lsum, citation precision, length, every ladder step against the previous one.

Secondary, all 948 questions: the same comparisons, plus MAUVE (whole-set). Set A + B were used by the divisions
to fix their settings, so the 948-question numbers are reported with that note.

Paired bootstrap: `critique/bootstrap.py`, 1,000 resamples, seed 0. Results are reported whatever they show.

## What had been seen before this plan was written

- Each component's own results above (all committed earlier by the divisions).
- While testing `combined/rerank_combined.py` on the GTR candidates (21 Nov 2026), the critique + grounding ranking
  on the GTR passages was evaluated on Set C: str-em +1.36 [+0.05, +2.66], citation recall +5.76 [+3.70, +7.86]
  vs C1. This row has no retrieval and is not R3. No ranking on the reranked passages beyond the retrieval
  division's own R1 run had been scored when this plan was written.

## Scoring

Per-candidate exact ALCE metrics for the reranked-passage candidates are computed once on the GPU
(`cluster/run_combined_prep.sbatch`, same procedure as the grounding division's `run_cand_exact_cit.sbatch`), so
every ranking is scored by lookup (`grounding/lookup_perq.py`). The final R3 answers for all 948 questions are also
scored directly with `cluster/run_critique_score.sbatch --mauve`, which checks the lookup and gives MAUVE.
