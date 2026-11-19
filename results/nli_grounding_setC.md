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

## Also reported (pre-registered, no claim attached)

### Changed picks vs changed answer texts

D2 changes 365 picks, 346 with a different answer text; D2-v2 changes 224 picks, 213 with a different text (the rest
pick another candidate whose output text is identical). Same table layout as for Set B
(`grounding/changed_breakdown.py`, outputs `results/runs/changed_ground_C.json`, `changed_ground_v2_C.json`;
the script reproduces the Set B table exactly).

### On the questions whose answer text changed

| On the changed questions | Released | D2 (346) | Released | D2-v2 (213) |
| --- | --- | --- | --- | --- |
| Citation recall | 58.8 | **71.4** | 54.7 | **70.1** |
| Fully supported (citation 100) | 171 | 222 | 98 | 123 |
| str-em | 30.2 | 25.6 | 27.0 | 31.0 |
| ROUGE-Lsum | 36.6 | 34.3 | 34.4 | 36.3 |
| Sentences | 1.70 | 1.61 | 1.66 | 1.73 |

Per question, new vs released (better / worse; the rest tie):

| | D2 | D2-v2 |
| --- | --- | --- |
| Citation recall | 119 / 61 | 78 / 38 |
| str-em | 54 / **89** | 44 / 32 |
| ROUGE-Lsum | 148 / **189** | 108 / 97 |

Reading: on Set C, as on Set B, D2's swaps are better grounded (+51 fully supported answers) but lose more often
than they win on str-em and ROUGE. With the relevance gate, D2-v2 keeps most of the grounding gain (+25 fully
supported on 213 swaps) and its swaps win slightly more often than they lose on str-em and ROUGE.

### Extra seconds per question for DeBERTa scoring of 5 candidates

Measured on one TC2 L40S (job 56136, `grounding/time_ground.py`, `cluster/run_time_ground.sbatch`; output
`results/runs/time_ground_C.json`), in the production setting (fp32, slow tokenizer, whole own passage as premise,
max 512 tokens), model load excluded. Set C has 9,444 NLI inputs, i.e. 12.6 (passage, sentence) pairs per question
for the 5 candidates. The re-run reproduces the stored p_ent (max |Δ| = 2.0e-5 over 9,444 inputs).

| Mode | Seconds per question |
| --- | --- |
| Batched over all of Set C (batch 16, offline) | 0.094 (70.5 s for 748 questions) |
| One question at a time (its 5 candidates in one batch, online) | mean 0.094, median 0.084, p90 0.142, max 0.446 |

So D2 adds about 0.1 s per question on an L40S, on top of Self-RAG's own generation. D2-v2 also needs the bge
relevance score of 5 (question, passage) pairs, a smaller cross-encoder; it was not timed (not pre-registered).

## Files

`results/runs/rr_ground_C.json`, `rr_ground_v2_C.json` (+ `.config.json`, `.perq.jsonl`, `.bootstrap.json`);
baseline `rr_released_lookup_C.json` (+ `_v2`).
