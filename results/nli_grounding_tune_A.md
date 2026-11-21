# NLI grounding (D2), Step 3: tuning on Set A and the fixed setting

Set A only (100 questions). Set B and C have not been re-ranked or scored with any D2 setting.

**Fixed setting: mode `mix`, aggregation `mean`, α = 0.75** (`grounding/configs/d2_final.json`)

```
support := 0.25 * ISSUP + 0.75 * g_mean        score = released - ISSUP + support
g_mean  = mean over the candidate's ALCE sentences of DeBERTa P(entailment | candidate's own passage)
```

## Code

| File | What |
| --- | --- |
| `grounding/rerank_ground.py` | Re-picks answers with the grounding term (modes `add` / `mix`) and writes an ALCE-format file + `.config.json`. Parameter 0 reproduces the released answers **200/200, identical text** (A+B, both modes). |
| `grounding/tune_ground.py` | Set A grid by table lookup in `cand_metrics_AB_exact.jsonl` (exact ALCE metrics per candidate); refuses other splits. Its choices for the fixed setting equal `rerank_ground.py`'s on 100/100 questions. |

## Selection rule (written into `tune_ground.py` before the grid was run)

1. Eligible: Set A str-em difference vs released ≥ −0.5 (the team's noise threshold).
2. c* = best citation-recall gain among eligible `mean` settings; plateau = eligible settings within 1.0 of c*.
3. Mode with more plateau members (tie → `add`); parameter = median of that mode's plateau values.
4. `soft_mean` rows are robustness only, never selected.

Result: 6 settings eligible, c* = +9.83, plateau = {mix α = 0.75} → **mix, α = 0.75**.

## Set A grid (differences vs released, 95% paired bootstrap)

| agg | mode | param | changed | str-em | citation rec | citation prec | ROUGE-L | length |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| — | released | — | 0 | 27.78 | 67.22 | 67.22 | 33.20 | 28.0 |
| mean | add | 0.1 | 20 | −0.73 [−2.4, 1.0] | +2.42 [−1.5, 6.7] | +2.42 | +0.33 | 27.9 |
| mean | add | 0.25 | 37 | −0.27 [−2.7, 2.6] | +5.42 [−0.6, 12.0] | +5.42 | −0.63 | 26.9 |
| mean | add | 0.5 | 40 | −1.27 [−4.5, 2.1] | +6.00 [−0.7, 12.8] | +6.00 | −0.85 | 26.2 |
| mean | add | 0.75 | 46 | +0.73 [−3.5, 5.2] | +8.50 [1.2, 15.8] | +8.50 | −1.13 | 26.2 |
| mean | add | 1.0 | 49 | −0.27 [−4.6, 4.6] | +8.00 [0.7, 15.7] | +8.00 | −1.48 | 26.7 |
| mean | mix | 0.25 | 37 | −0.27 [−2.7, 2.6] | +5.42 [−0.6, 12.0] | +5.42 | −0.50 | 26.8 |
| mean | mix | 0.5 | 47 | +0.73 [−3.3, 5.3] | +8.00 [0.7, 15.7] | +8.00 | −1.25 | 26.8 |
| **mean** | **mix** | **0.75** | **52** | **−0.02 [−4.8, 5.2]** | **+9.83 [2.3, 17.8]** | **+9.83 [2.0, 18.2]** | **−1.15 [−2.9, 0.5]** | **26.1** |
| mean | mix | 1.0 (= replace) | 59 | −0.88 [−6.0, 4.7] | +10.03 [1.8, 18.3] | +10.03 | −1.68 | 26.1 |
| soft_mean | add | 0.75 | 42 | −0.47 [−2.9, 2.5] | +6.67 [0.3, 13.7] | +6.67 | −0.54 | 26.7 |
| soft_mean | mix | 0.75 | 52 | +0.03 [−4.5, 4.6] | +10.00 [2.2, 17.7] | +10.00 | −1.06 | 26.7 |

(All rows: `results/ground_tune_A.json`.)

## Reading

- **Citation recall/precision +9.8 on Set A with str-em unchanged (−0.02).** That is 60% of the T5 upper bound
  (+16.3, `nli_grounding_step05.md`) with a checker that is not the grader.
- **The shape is smooth.** Every `mean` setting from add 0.75 / mix 0.5 upward gives +8 to +10 citation; str-em
  stays inside ±1.3 throughout. The chosen α = 0.75 sits between mix 0.5 (+8.0) and mix 1.0 (+10.0, but str-em
  −0.88, ineligible), so it is the middle of that region rather than an isolated peak, even though the
  1-point plateau rule left it as the only member.
- **`soft_mean` gives the same picture** (mix 0.75: +10.0 citation, +0.03 str-em). The result does not hinge on
  how neutral is treated.
- **Costs to watch on Set B / C:** ROUGE-L −1.15 [−2.9, +0.5] and answers about 2 words shorter (26.1 vs 28.0).
  Grounding prefers candidates with fewer unsupported extra sentences (1.44 vs 1.56 sentences on the changed
  questions).
- **Many questions change (52, of which 50 change the answer text).** Released scores of the 5 candidates are often
  close (relevance and ISSUP both near 1), so the grounding term decides many near-ties. Effects on Set B will
  still be noisy; Set C is the decisive test.
- Compared with sy's D4 (Set B citation −7.5): D2 moves citation in the opposite direction on Set A, as predicted
  by the diagnostic that the answers D4 swapped in were poorly grounded.

## Next (Step 4, Set B)

Commit this file, `grounding/configs/d2_final.json` and the code **before** running Set B. Then:

```bash
python grounding/rerank_ground.py --f results/runs/asqa_cands_0_end.json --ground results/runs/cand_ground_B.jsonl \
    --split critique/splits/asqa_B.json --mode mix --agg mean --param 0.75 --out out/rr_ground_B.json
# exact metrics: score_per_question.py on the cluster (or table lookup in cand_metrics_AB_exact.jsonl, which is
# identical to it by construction), then
python critique/bootstrap.py --base results/runs/rr_released.json.perq.jsonl \
    --new out/rr_ground_B.json.perq.jsonl --split critique/splits/asqa_B.json
```
