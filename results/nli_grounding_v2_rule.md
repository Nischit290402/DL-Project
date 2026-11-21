# D2-v2 (relevance-gated NLI grounding): setting and selection rule, written before the grid is run

Status: written 2026-11-19, **before** `grounding/tune_ground_v2.py` was run. Set C has not been scored or
looked at for D2 or D2-v2. The original D2 (`grounding/configs/d2_final.json`) stays the pre-registered primary
Set C test; D2-v2 is a clearly labelled exploratory variant.

## Why a variant, and why A+B is the development set

Set B showed D2 lowers ROUGE-L significantly (−2.88 [−5.41, −0.52]). The error analysis of all 90 A+B swaps
(`results/nli_grounding_labels_and_probe.md`) found most bad swaps move to a passage that does not answer the
question. A reranker (BAAI/bge-reranker-base) separated good from bad swaps (AUC 0.76). Because Set B informed
this, **A and B together (200 questions) are the development set for D2-v2**, and Set C is its only clean test.

## The variant

Unchanged from D2: candidates, `g` = mean DeBERTa p_ent over the answer's sentences (whole own passage as
premise, `cand_ground_*.jsonl`), mix score `released + α·(g − ISSUP)`.

New: a relevance gate. For each question, let `r(c)` = bge-reranker-base logit of (question, "title. text") for
candidate c's passage (`grounding/qa_relevance.py`), and `r0` = r of the released choice. Only candidates with
`r(c) ≥ r0 − δ` are eligible; D2-v2 picks the eligible candidate with the highest mix score (same tie and
empty-answer handling as D2). The released choice is always eligible, so δ → ∞ gives D2 back.

## Grid (9 settings), on A+B, by exact lookup in `cand_metrics_AB_exact.jsonl`

α ∈ {0.5, 0.75, 1.0} × δ ∈ {0, 1, 2}. Ungated D2 rows (α = 0.5, 0.75, 1.0) are printed for reference only.

## Selection rule

Differences vs released, paired over the A+B questions (point estimates; 95% bootstrap intervals reported):

1. Eligible: Δstr-em ≥ −0.5 **and** ΔROUGE-L ≥ −0.5 (the same noise threshold, now also for ROUGE).
2. c* = the best Δcitation recall among eligible settings; plateau = eligible settings with Δcitation ≥ c* − 1.0.
3. From the plateau, take the setting nearest the grid centre (α = 0.75, δ = 1), counting distance in grid steps
   |Δα steps| + |Δδ steps|; ties go to the smaller δ (stricter gate), then the smaller α.
4. If nothing is eligible: the setting with the best Δcitation among those with Δstr-em ≥ −0.5 and
   ΔROUGE-L ≥ −1.0; if there is still none, D2-v2 is dropped and only D2 goes to Set C.

The chosen setting is written to `grounding/configs/d2v2_final.json` and frozen before any Set C run.

## Set C plan (unchanged for D2, added for D2-v2)

Run once each: D2 (`d2_final.json`) and D2-v2 (`d2v2_final.json`), paired 95% bootstrap vs released.
D2: primary = citation recall gain; secondary = str-em and ROUGE-L non-inferiority. D2-v2: same metrics,
reported as exploratory (designed with Set B in view).
