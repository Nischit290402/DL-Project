# D2 NLI grounding: Set C results (pre-registered in `grounding/PREREG_C.md`, commit `ef03850`)

Run once each, 2026-11-19, with the pre-registered commands. All 8 frozen inputs matched their sha256 at HEAD
`3355984` (= `origin/nli-grounding`). One deviation, recorded before any D2 run (`e898799`,
`grounding/PREREG_C_deviations.md`): the multi-segment question q471 is not in the lookup table, so its row is taken
from the GPU scoring of the released answer, which every system keeps unchanged (Δ = 0 on q471).

## Sanity checks

1. `--param 0` reproduces the released choice: 748/748, answer changed 0 (also with `--rel … --delta 1`).
2. Candidates without a grounding score: 0; no KeyError in the relevance gate.
3. Released lookup vs sy's GPU scoring: 0 of 747 questions differ (q471 not in the table, see deviation).

## Results (748 questions, paired bootstrap, 1,000 resamples, seed 0; Δ = system − released)

| Metric | Released | D2 | Δ D2 [95%] | D2-v2 | Δ D2-v2 [95%] |
| --- | --- | --- | --- | --- | --- |
| str-em | 30.02 | 27.91 | **−2.11 [−3.82, −0.52]** | 31.15 | **+1.13 [+0.01, +2.28]** |
| ROUGE-Lsum | 35.89 | 34.83 | **−1.05 [−1.90, −0.22]** | 36.43 | +0.54 [−0.10, +1.23] |
| Citation recall | 65.93 | 71.75 | **+5.81 [+3.20, +8.48]** | 70.32 | **+4.39 [+2.31, +6.34]** |
| Citation precision | 66.94 | 72.31 | **+5.37 [+2.73, +8.25]** | 71.17 | **+4.23 [+2.29, +6.38]** |
| Length | 30.58 | 29.18 | −1.40 [−2.86, +0.07] | 30.84 | +0.26 [−0.86, +1.41] |

Bold = interval excludes 0. Answers changed vs released: D2 365 / 748, D2-v2 224 / 748.

## Verdicts by the pre-registered rules

- **H1 (D2 raises citation recall): confirmed.** +5.81 [+3.20, +8.48], close to Set B (+5.98).
- **H2a (str-em non-inferior, lower end ≥ −2.0): not met** (lower end −3.82; str-em significantly *down*).
- **H2b (ROUGE-Lsum non-inferior, lower end ≥ −1.0): not met** (lower end −1.90).
- **H3 (D2 lowers ROUGE-Lsum): confirmed.** −1.05 [−1.90, −0.22], smaller than Set B (−2.88).
- **Overall H-D2: trade-off.** Citation up, answer quality (str-em and ROUGE) down.

**D2-v2 (exploratory): consistent with development.** The relevance gate removes D2's ROUGE loss (lower end −0.10
≥ −1.0) and keeps a citation gain (+4.39, interval above 0); str-em is slightly up (+1.13 [+0.01, +2.28]), as on
A+B (+3.29). This does not change the D2 verdict, and D2-v2 was designed with Set B in view, so it is a held-out
replication of a development finding, not a confirmatory test.

## Not yet done (pre-registered "also reported", no claim attached)

Changed-question breakdown (released vs new citation recall, str-em, ROUGE, share fully supported), changed answer
*texts* vs changed picks, and seconds per question for DeBERTa scoring of 5 candidates.

## Files

`results/runs/rr_ground_C.json`, `rr_ground_v2_C.json` (+ `.config.json`, `.perq.jsonl`, `.bootstrap.json`);
baseline `rr_released_lookup_C.json` (+ `_v2`).
