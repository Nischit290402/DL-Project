# Deviations from `grounding/PREREG_C.md` (written 2026-11-19, after sanity checks 1-3, before any D2 output on Set C)

## 1. q471 (the one multi-segment question) is not in the candidate table

**What happened.** `rerank_ground.py` keeps the one multi-segment Set C question (q_idx 471) as is in every
setting ("1 multi-segment kept"; noted in `results/nli_grounding_step05.md`). It has no depth-1 candidate, so it
is not in `cand_metrics_C_exact.jsonl`, and `lookup_perq.py` scores 747 of the 748 questions. Two pre-registered
commands then fail as written: sanity check 3 (`KeyError: 471`, it iterates over the 748 GPU-scored questions) and
`bootstrap.py --split critique/splits/asqa_C.json` (asserts all 748 split questions are present). Sets A and B had
no multi-segment question, so this did not come up before.

**Sanity check 3 result** (run on the 747 questions in both files): released lookup vs GPU scoring, questions
differing = **0 of 747**. q471's released answer text in the lookup file is identical to sy's `rr_released_C.json`.

**Fix (decided before running D2).** Every system keeps q471's released answer unchanged, so its per-question
scores are the same for all of them. The q471 row from sy's GPU scoring (`results/runs/rr_released_C.json.perq.jsonl`)
is appended to each lookup `.perq.jsonl` (released, D2, D2-v2), after asserting that the system's q471 output text
equals the released one. This gives the 748 questions the pre-registration specifies and lets the pre-registered
bootstrap commands run unchanged. q471 contributes Δ = 0 to every difference. Nothing else changes.

Command, run on each lookup `.perq.jsonl` right after `lookup_perq.py`:

```bash
python grounding/fill_multiseg.py --f <rerank file> --perq <rerank file>.perq.jsonl \
    --released results/runs/rr_released_C.json --released_perq results/runs/rr_released_C.json.perq.jsonl
```
