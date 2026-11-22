# D2-v2 (relevance-gated NLI grounding): A+B development result, setting frozen for Set C

Rule written before the grid ran: `results/nli_grounding_v2_rule.md` (sha256 `fd02b13a…`, stored in the output).
Grid output: `results/ground_tune_v2_AB.json`. Frozen setting: `grounding/configs/d2v2_final.json`.
**A+B is the development set** (Set B informed the design); Set C, which has not been looked at, is the only clean test.

## What changed vs D2

Same grounding score and mix formula (α·(g − ISSUP)). New: a candidate may be chosen only if its passage is at
most δ less relevant to the question than the released choice's passage, by BAAI/bge-reranker-base
(`grounding/qa_relevance.py`, `grounding/rerank_ground.py --rel … --delta …`).

## Grid on A+B (200 questions; paired 95% bootstrap vs released)

| α | δ | changed | Δstr-em | ΔROUGE-L | Δcitation rec |
| --- | --- | --- | --- | --- | --- |
| 0.75 | no gate (= D2) | 97 | −0.13 [−3.62, 3.42] | **−2.02 [−3.50, −0.54]** | +7.91 [2.58, 13.25] |
| 0.5 | 0 | 46 | +2.83 [0.62, 5.29] | +0.48 [−0.42, 1.46] | +4.67 [1.00, 8.75] |
| 0.5 | 1 | 54 | +2.88 [0.42, 5.71] | 0.00 [−1.11, 1.08] | +5.66 [1.75, 9.75] |
| 0.5 | 2 | 65 | +2.17 [−0.62, 5.21] | −0.28 [−1.48, 0.92] | +6.58 [2.41, 10.79] |
| 0.75 | 0 | 50 | +3.08 [0.79, 5.62] | +0.32 [−0.69, 1.45] | +5.67 [1.71, 9.96] |
| **0.75** | **1** | **58** | **+3.29 [0.79, 6.12]** | **−0.17 [−1.36, 1.07]** | **+6.16 [2.15, 10.67]** |
| 0.75 | 2 | 71 | +3.08 [0.17, 6.12] | −0.37 [−1.65, 0.94] | +7.08 [2.79, 11.73] |
| 1.0 | 0 | 60 | +2.73 [0.40, 5.33] | +0.30 [−0.73, 1.43] | +4.67 [0.37, 9.04] |
| 1.0 | 1 | 69 | +2.69 [0.02, 5.73] | −0.26 [−1.45, 0.98] | +4.99 [0.62, 9.71] |
| 1.0 | 2 | 80 | +2.48 [−0.52, 5.58] | −0.45 [−1.76, 0.89] | +5.91 [1.42, 10.67] |

Selection (rule applied mechanically): all 9 settings eligible; best citation +7.08; plateau
{(0.5, 2), (0.75, 1), (0.75, 2)}; nearest the grid centre → **α = 0.75, δ = 1**.

Per set with `rerank_ground.py` (choices identical to the tuning lookup on 200/200 questions):

| Set | Δstr-em | ΔROUGE-L | Δcitation rec |
| --- | --- | --- | --- |
| A | +4.17 [+0.83, +8.17] * | +0.23 [−1.05, +1.52] | +8.17 [+2.00, +15.17] * |
| B | +2.42 [−1.17, +6.33] | −0.57 [−2.56, +1.55] | +4.15 [−1.68, +9.90] |

## Reading

- Every gated setting removes D2's significant ROUGE loss (all intervals now include 0) and turns str-em
  positive (7 of 9 intervals exclude 0). The price is a smaller citation gain (+6.2 vs +7.9).
- The result is smooth across α and δ, so it does not hinge on the exact setting.
- The str-em gain is plausible but optimistic: Set B shaped the idea, and A+B selected δ. Set C will tell.
- Set C relevance scores (`results/runs/cand_qarel_C.jsonl`, 3,740 candidates) are computed but unused.

## Set C (next, once each)

```bash
python grounding/rerank_ground.py --f results/runs/asqa_cands_C.json --ground results/runs/cand_ground_C.jsonl \
    --split critique/splits/asqa_C.json --mode mix --agg mean --param 0.75 --out out/rr_ground_C.json
python grounding/rerank_ground.py --f results/runs/asqa_cands_C.json --ground results/runs/cand_ground_C.jsonl \
    --split critique/splits/asqa_C.json --mode mix --agg mean --param 0.75 \
    --rel results/runs/cand_qarel_C.jsonl --delta 1 --out out/rr_ground_v2_C.json
python grounding/lookup_perq.py --f out/rr_ground_C.json --metrics results/runs/cand_metrics_C_exact.jsonl
python grounding/lookup_perq.py --f out/rr_ground_v2_C.json --metrics results/runs/cand_metrics_C_exact.jsonl
python critique/bootstrap.py --base results/runs/rr_released_C.json.perq.jsonl --new out/rr_ground_C.json.perq.jsonl
python critique/bootstrap.py --base results/runs/rr_released_C.json.perq.jsonl --new out/rr_ground_v2_C.json.perq.jsonl
```
D2 is the pre-registered primary test; D2-v2 is reported as exploratory.
