# NLI grounding (D2), Step 0.5 + Step 1: per-candidate metrics, ceilings, DeBERTa grounding scores

Branch `nli-grounding` (from `self-contradiction`). Revised 2026-11-19 with **exact** per-candidate ALCE citation
(T5-11B, TC2 L40S, job 55865); the earlier proxy-based numbers are superseded.
Set C candidate metrics and grounding scores are computed and saved but **not analysed** (kept for the
pre-registered test).

## 1. Per-candidate metrics

For each of the 5 depth-1 candidates per question, the answer text Self-RAG would output if that candidate were
chosen (`critique/rerank.py` `build_output`) is scored exactly like `critique/score_per_question.py`.

| File | Candidates | What | How |
| --- | --- | --- | --- |
| `results/runs/cand_metrics_AB.jsonl` | 1000 (A+B) | str-em, ROUGE-Lsum, length, `cit_proxy` | `grounding/cand_metrics.py`, CPU |
| `results/runs/cand_metrics_C.jsonl` | 3740 (C) | same, no proxy | CPU |
| **`results/runs/cand_metrics_AB_exact.jsonl`** | 1000 / 1000 | + exact `citation_rec`, `citation_prec` | T5-11B, `cluster/run_cand_exact_cit.sbatch` |
| **`results/runs/cand_metrics_C_exact.jsonl`** (not analysed) | 3735 / 3740 | same | same job |

**Use the `_exact` files for all citation numbers.**

Validation:
- str-em, length and ROUGE-Lsum computed on CPU equal the GPU-scored released answers on **200/200 (A+B) and
  747/747 (C)**, and equal the T5 job's own str-em/ROUGE for all candidates (0 mismatches).
- Exact citation of the selected candidates equals the released per-question citation recall on **200/200 and
  747/747**.
- The one multi-segment question in C has no depth-1 choice and is skipped (5 candidates).

The exact job: `grounding/make_pick_files.py` writes 5 files per set (file k = every question answers with its
k-th candidate), `critique/score_per_question.py` scores each, `grounding/merge_exact_cit.py` merges and runs the
checks. 23 min on TC2N07. Log: `results/runs/cand_exact_cit_55865.log`.

**Why the proxy was not enough (kept as a finding).** Self-RAG's `[IsSup]` grade, and hence the saved T5 verdicts,
covers only the text before the first grade, usually the first sentence; **36% of A+B candidates have 2+
sentences that ISSUP never grades**. The proxy (mean verdict over graded sentences) was exact where every sentence
was graded (128/128 chosen answers) and biased upward otherwise (+3.9 / +3.2 points on average over A / B
candidates).

Mean citation recall by passage slot (A+B, exact): 60.7 / 47.1 / 36.2 / 36.8 / 36.3 for slots 0–4. The
top-retrieved passage gives by far the best-grounded candidate.

## 2. Ceilings (`grounding/ceilings.py` → `results/ground_ceilings_AB.json`)

Exact ALCE citation. Differences are paired against the released choice (C1), 95% bootstrap interval, 2000
resamples. Set B is shown for scale only; no setting is chosen from it.

| | A str-em | A cit. rec | A cit. prec | B str-em | B cit. rec | B cit. prec | changed A / B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Released (C1) | 27.78 | 67.22 | 67.22 | 33.92 | 70.35 | 71.67 | 0 / 0 |
| Oracle, best str-em | **+12.40** [8.1, 17.3] | −6.20 | −6.20 | **+15.27** [10.8, 20.2] | −2.95 | −3.45 | 28 / 30 |
| Oracle, best citation | −1.37 [−5.3, 2.7] | **+20.70** [14.0, 28.7] | +20.70 | +2.02 [−2.0, 6.2] | **+14.32** [8.6, 21.2] | +13.50 | 28 / 20 |
| Oracle, best citation, then best str-em | **+5.42** [0.6, 10.6] | **+20.70** | +20.70 | **+7.52** [2.7, 12.8] | **+14.32** | +13.50 | 39 / 32 |
| T5 verdict replaces ISSUP (upper bound, not a method) | +0.33 [−4.2, 4.8] | **+16.28** [9.2, 23.8] | +16.28 | +2.43 [−1.6, 6.4] | **+11.17** [5.2, 17.7] | +10.50 | 36 / 32 |
| Released + 0.25 × T5 verdict | −0.83 [−4.0, 2.0] | +12.70 [6.5, 19.6] | +12.70 | +2.02 [−1.0, 5.5] | +10.50 [5.2, 16.2] | +9.83 | 14 / 13 |
| Released + 0.5 × T5 verdict | −0.83 | +14.70 | +14.70 | +2.52 | +11.50 | +10.83 | 16 / 14 |

Headroom: candidates differ in citation recall on 88 / 87 questions (A / B) and in str-em on 67 / 77. The released
answer is not the citation-best candidate on 28 / 20 questions and not the str-em-best on 28 / 30. Mean candidate
citation recall is 43.7 / 43.1, so the released choice (67 / 70) is already far above a random pick.

Within-question concordance (pairs of candidates of one question that differ on the target; 0.5 = no signal):

| Signal | str-em, A (365 pairs) | str-em, B (422) | citation rec, A (527) | citation rec, B (503) |
| --- | --- | --- | --- | --- |
| Released score | 0.682 (= sy's D2, cross-check) | 0.671 | 0.751 | 0.761 |
| ISSUP `ground_score` alone | 0.677 | 0.665 | 0.757 | 0.767 |
| T5 verdict (graded claims only) | 0.577 | 0.626 | **0.893** | **0.872** |

## 3. What this means for D2

1. **Citation headroom is large and real (exact).** A perfect passage-grounding checker (T5 on the graded claim)
   adds **+16.3 / +11.2** citation recall (A / B) with str-em flat. Adding it with a small weight (μ = 0.25)
   already gives +12.7 / +10.5 while changing only 14 / 13 answers. This row ranks with the grader's own model, so
   it is inflated by construction (grader leakage): an upper bound, not a method.
2. **Citation and correctness do not trade off inherently.** Picking the best-cited candidate and breaking ties by
   str-em raises both (+20.7 citation, +5.4 str-em on A). So "citation up, str-em not down" (H-D2) is achievable
   in principle; the str-em loss of sy's D4 came from its signal, not from the task.
3. **ISSUP is good at str-em ordering, mediocre at citation ordering; an outside NLI checker is the reverse.**
   Released/ISSUP: 0.68 str-em vs 0.75–0.77 citation; T5: 0.58–0.63 str-em vs 0.87–0.89 citation. D2 should
   *add* grounding to the released score (modes b/c), not replace ISSUP (mode a).
4. **Decision point (plan v3, Step 0.5): go.** Far above the "< 2 points" stop line.
5. **Expected effect size on Set B is about 10–20 changed answers out of 100.** Intervals will be wide, so the
   decisive test is the pre-registered Set C (748 questions), whose exact per-candidate citation now exists too:
   any Set C configuration can be evaluated by table lookup, with no further GPU run.

## 4. Step 1: DeBERTa grounding scores (`grounding/cand_ground.py`)

Premise = the candidate's own passage (`Title: <t>\n<text>`), hypothesis = each ALCE sentence of the answer the
candidate would give (same split as ALCE citation recall), plus one whole-answer row (`sent_idx = -1`).
`cross-encoder/nli-deberta-v3-large`, fp32, labels read from `id2label` ({0: contradiction, 1: entailment, 2: neutral}).

| File | Rows | Candidates | Time (RTX 3060 Laptop, fp32) |
| --- | --- | --- | --- |
| `results/runs/cand_ground_A.jsonl` | 1266 | 500 | 34 s |
| `results/runs/cand_ground_B.jsonl` | 1242 | 500 | 35 s |
| `results/runs/cand_ground_C.jsonl` | 9444 | 3740 (not analysed) | 271 s |

About 0.35 s per question for all 5 candidates (C1 generation: 2.54 s/question on L40S). No input reached 512
tokens. Every candidate has rows, and the per-candidate sentence count equals `cit_sents` for all 4740 candidates.
These runs are on a laptop GPU, not L40S; if the team requires L40S for every number, rerun with
`sbatch cluster/run_cand_ground.sbatch` (no generation involved, expected to differ only in the last decimals).

Diagnostics G1–G4 on Set A: `results/nli_grounding_diag_A.md`.

### Local environment (Windows, everything on E:)

```bash
python -m venv E:/Anaconda_Envs/dlpj_d2
E:/Anaconda_Envs/dlpj_d2/Scripts/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
E:/Anaconda_Envs/dlpj_d2/Scripts/python -m pip install transformers==4.46.3 sentencepiece protobuf nltk rouge_score
# before running: HF_HOME=E:\Anaconda_Envs\hf_cache  NLTK_DATA=E:\Anaconda_Envs\nltk_data  TMP/TEMP=E:\Anaconda_Envs\_tmp
```

## Commands

```bash
# CPU
python grounding/cand_metrics.py --f results/runs/asqa_cands_0_end.json \
    --verdicts results/runs/asqa_cands_0_end.json.verdicts.jsonl --out results/runs/cand_metrics_AB.jsonl
python grounding/cand_metrics.py --f results/runs/asqa_cands_C.json --out results/runs/cand_metrics_C.jsonl
# GPU (TC2, T5-11B): exact citation for every candidate, A+B and C
sbatch cluster/run_cand_exact_cit.sbatch
# CPU
python grounding/ceilings.py --metrics results/runs/cand_metrics_AB_exact.jsonl --out results/ground_ceilings_AB.json
# GPU (any; DeBERTa)
for S in A B; do python grounding/cand_ground.py --f results/runs/asqa_cands_0_end.json \
    --split critique/splits/asqa_$S.json --out results/runs/cand_ground_$S.jsonl; done
python grounding/cand_ground.py --f results/runs/asqa_cands_C.json \
    --split critique/splits/asqa_C.json --out results/runs/cand_ground_C.jsonl
```
