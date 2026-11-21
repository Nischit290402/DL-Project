# Pre-registered tests on Set C (written 2026-11-18, before any Set C output exists)

Set C = the 748 ASQA dev questions in neither Set A nor Set B (`critique/splits/asqa_C.json`). None of them has
been used by the self-contradiction work: no candidates, NLI scores or re-picked answers exist for them when this
file is committed. Everything below is fixed now and will be reported whatever it shows. No other variant, threshold
or lambda will be tried on Set C.

## Data produced (one job, `cluster/run_contra_C.sbatch`)

1. Candidates for Set C: `run_long_form_static.py` with the C1 settings plus `--save_candidates --indices_file
   critique/splits/asqa_C.json` -> `results/runs/asqa_cands_C.json` (L40S).
2. NLI over every ordered pair of depth-1 candidates (`contradiction/cand_nli.py`) -> `results/runs/cand_nli_C.jsonl`.
3. Released choices (`rerank_contra.py --lam 0`) -> `rr_released_C.json`; penalty choices (defaults: noctx,
   lambda 0.3, fixed on Set A on 2026-11-17) -> `rr_contra_C.json`.
4. Per-question ALCE scores for both (`critique/score_per_question.py`).

## H1 (primary): disagreement among candidates flags less correct answers

Disagreement D_q = mean over the 10 candidate pairs of max(p_con(i->j), p_con(j->i)), noctx. Split Set C at the
median of D_q. Statistic: mean str-em of the released answer in the low-disagreement half minus the
high-disagreement half, 95% bootstrap interval (2000 resamples of questions, seed 6103, median split redone in each).

- **Confirmed if the interval lies above 0.** Not confirmed otherwise.
- Expected from Sets A + B (exploratory, where the idea came from): +9.67 [+1.72, +17.90].
- Reported alongside, with no claim attached: the same split for citation recall and ROUGE-Lsum, and the AUROC of
  D_q for "released str-em = 0" with its interval.
- Command: `contradiction/uncertainty_test.py` (exact call in the job script).

## H2: the contradiction penalty on held-out questions

Penalty choices vs released choices on Set C, paired bootstrap (`critique/bootstrap.py`).

- H2a, predicted from Set B: the penalty does not improve str-em, i.e. the interval of the difference does not lie
  above 0.
- H2b, predicted from Set B: the penalty lowers citation recall, i.e. the interval of the difference lies below 0.
- Each is reported as confirmed or not confirmed by those rules.

## Sanity check (not a hypothesis)

Released choices regenerated for Set C vs C1 (`results/runs/asqa_keeptok_0_end.json`) on the same questions: same
settings and GPU model, so differences should be near 0. A large gap would mean the candidate run differs from C1 and
is reported as a caveat.
