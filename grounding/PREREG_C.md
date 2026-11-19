# D2 NLI grounding: pre-registered tests on Set C (written 2026-11-19, before any D2 output on Set C exists)

Set C = the 748 ASQA dev questions in neither Set A nor Set B (`critique/splits/asqa_C.json`). Everything below is
fixed when this file is committed and pushed, and will be reported whatever it shows. No other mode, aggregation,
α, δ, relevance model or NLI premise will be run on Set C. Each of the two settings below is run exactly once.

## What Set C has and has not been used for

- **Used before, for other hypotheses:** sy's self-contradiction tests (`contradiction/PREREG_C.md`, commit
  `f94597b`, results `a90ccbc`) generated the Set C candidates and reported released vs contradiction-penalty
  scores on Set C. That work tested a different signal (disagreement *between* candidates). D2 was designed on
  Set A (D2) and Set A+B (D2-v2) without looking at any Set C metric.
- **Computed for D2 but not analysed** (only line counts and coverage checked):
  `cand_ground_C.jsonl` (DeBERTa grounding, 9,444 rows), `cand_qarel_C.jsonl` (bge relevance, 3,740 candidates),
  `cand_metrics_C_exact.jsonl` (exact per-candidate ALCE metrics, 3,740 candidates, T5-11B job 55865). No D2
  re-ranked file, ceiling, oracle or per-candidate statistic has been computed on Set C.
- Critique P2 (`origin/critique-p2`, 82d0ca9) did not use Set C.

## Frozen inputs (sha256 of the committed, LF-normalised files)

| File | sha256 |
| --- | --- |
| `grounding/configs/d2_final.json` (D2, fixed on Set A in `6e3b386`) | `9e9a30866fbba441f41a041b0dadf5159b6f2345bfb4e5b37d45cee7b1f6ec5f` |
| `grounding/configs/d2v2_final.json` (D2-v2, fixed on Set A+B) | `a99cba721a4efc5b17cc2e625c64fd70b1a6805737c578961af77055f10feedf` |
| `results/nli_grounding_v2_rule.md` (D2-v2 selection rule) | `fd02b13ab0436b179a50e491f33b3af17d6f384e6ef7e3cd0c4585caa7df9a72` |
| `critique/splits/asqa_C.json` | `b38331e0b79acd70621c8a18007bfaa4156a1b5bf5c6cb02e90eb7af31fa6f62` |
| `results/runs/asqa_cands_C.json` (candidates, L40S, sy) | `ff1750ce492d2f4c2770ffe14a51107a26ba21812c41d237cdeb71be52616849` |
| `results/runs/cand_ground_C.jsonl` | `2305281386ece2e55ca27c3ea716ee2cd8ca5add30a79990481c8bd236e10085` |
| `results/runs/cand_qarel_C.jsonl` | `e1082fb9166da48df52c8e752f77c040d8981e9de3a52360370331b1b630e120` |
| `results/runs/cand_metrics_C_exact.jsonl` | `5605692464b9155605d1181d7b7697dc70fa14348643746a818ce52294f9143a` |

Check before running: `git show HEAD:<file> | sha256sum` for each row must match.

**Compute environment.** The DeBERTa grounding scores (`cand_ground.py`, `cross-encoder/nli-deberta-v3-large`,
fp32, slow tokenizer) were computed locally on an RTX 3060, not on the L40S nodes. A TC2 L40S re-run of 170 of
those sentences in the production setting (job 55962, `nli_probe.py`) reproduced the stored p_ent with
max |Δ| < 1e-4. They are not re-run. The candidates, the relevance scores and the exact ALCE metrics are all from
TC2 L40S. Re-ranking, lookup and bootstrap are CPU-only and deterministic.

## Settings under test

- **D2 (primary, confirmatory).** `mix`, `mean`, α = 0.75: score = released − ISSUP + (0.25·ISSUP + 0.75·g_mean),
  g_mean = mean DeBERTa P(entail | own passage) over the answer's ALCE sentences.
- **D2-v2 (secondary, exploratory).** Same score, but only candidates with bge-reranker-base relevance
  r(c) ≥ r(released choice) − 1.0 are eligible (released choice always eligible). Set B motivated this variant
  and A+B selected δ, so Set C is its only held-out test; it is reported as exploratory, not as a confirmation.

Baseline for both: the released (C1) choices on Set C.

## Analysis (fixed)

Per-question scores of every system, the baseline included, come from the same table
(`grounding/lookup_perq.py` on `cand_metrics_C_exact.jsonl`). Paired bootstrap: `critique/bootstrap.py` with its
defaults (1,000 resamples, seed 0), the same as for Set B. "Interval" = the 95% interval of the difference
(system − released) over the 748 questions.

## H1 (primary): D2 raises citation recall

- **Confirmed if the interval of Δcitation recall lies above 0.** Not confirmed otherwise.
- Expected from development: Set A +9.83 [+2.3, +17.8]; Set B +5.98 [−0.75, +12.88].

## H2 (secondary): D2 does not cost answer quality (non-inferiority)

- H2a, str-em: **non-inferior if the lower end of the Δstr-em interval is ≥ −2.0.** (Set B interval half-width
  ≈ 5; on 748 questions it is expected near 1.8, so −2.0 is the smallest margin the test can meet with a point
  estimate near 0.)
- H2b, ROUGE-Lsum: **non-inferior if the lower end of the ΔROUGE-Lsum interval is ≥ −1.0.**
- Each is reported as met or not met by those rules.

## H3 (predicted from Set B): D2 lowers ROUGE-Lsum

- Set B: −2.88 [−5.41, −0.52]. **Confirmed if the ΔROUGE-Lsum interval lies below 0.**
- H2b and H3 can both fail (interval straddles 0 with lower end < −1.0); then ROUGE is reported as inconclusive.

## Overall verdict on H-D2 ("citation up without hurting answer quality")

- **Supported** if H1 is confirmed and H2a and H2b are both met.
- **Trade-off** if H1 is confirmed but H2a or H2b is not met.
- **Not supported** if H1 is not confirmed.

## D2-v2 (exploratory, reported alongside, no confirmation claimed)

Same five metrics and intervals as D2. Reported as "consistent with development" or not, by comparing with its A+B
result: Δstr-em +3.29 [+0.79, +6.12], ΔROUGE-L −0.17 [−1.36, +1.07], Δcitation +6.16 [+2.15, +10.67]. The
specific question is whether the gate removes D2's ROUGE loss on Set C (D2-v2 ΔROUGE-L interval lower end ≥ −1.0)
while keeping a citation gain (interval above 0). Whatever D2-v2 shows, it does not change the D2 verdict.

## Also reported, with no claim attached

Length; citation precision; number of changed picks and changed answer texts; on the changed questions, released
vs new citation recall, str-em, ROUGE-Lsum and share of fully supported answers (same table as for Set B); extra
seconds per question for DeBERTa scoring of 5 candidates.

## Sanity checks (not hypotheses; run before reading any D2 result)

1. `rerank_ground.py --param 0` on Set C reproduces the released choice on 748/748 questions (and also with
   `--rel … --delta 1`, which only gates a param-0 score). Any mismatch stops the run until explained.
2. No candidate is missing from `cand_ground_C.jsonl` (the script prints "candidates without a grounding
   score: 0") or from `cand_qarel_C.jsonl` (a missing one raises a KeyError in the gate).
3. The lookup per-question scores of the released choices vs `results/runs/rr_released_C.json.perq.jsonl`
   (sy's GPU scoring): number of questions with any metric differing. Either way the lookup baseline is used, so
   both systems are scored by the same table; the count is reported as an environment note.

## Commands (run once, in this order, after this file is committed and pushed)

```bash
# sanity checks 1-3
python grounding/rerank_ground.py --f results/runs/asqa_cands_C.json --ground results/runs/cand_ground_C.jsonl \
    --split critique/splits/asqa_C.json --mode mix --agg mean --param 0 --out out/rr_released_lookup_C.json
python grounding/rerank_ground.py --f results/runs/asqa_cands_C.json --ground results/runs/cand_ground_C.jsonl \
    --split critique/splits/asqa_C.json --mode mix --agg mean --param 0 \
    --rel results/runs/cand_qarel_C.jsonl --delta 1 --out out/rr_released_lookup_v2_C.json
python grounding/lookup_perq.py --f out/rr_released_lookup_C.json --metrics results/runs/cand_metrics_C_exact.jsonl
python -c "import json; L=lambda p: {r['q_idx']: r for r in map(json.loads, open(p))}; \
a=L('out/rr_released_lookup_C.json.perq.jsonl'); b=L('results/runs/rr_released_C.json.perq.jsonl'); \
K=('str_em','rougeLsum','citation_rec','citation_prec','length'); \
print('released lookup vs GPU scoring: questions differing =', sum(any(abs(a[q][k]-b[q][k])>1e-6 for k in K) for q in b), 'of', len(b))"

# D2 (primary) and D2-v2 (exploratory)
python grounding/rerank_ground.py --f results/runs/asqa_cands_C.json --ground results/runs/cand_ground_C.jsonl \
    --split critique/splits/asqa_C.json --mode mix --agg mean --param 0.75 --out out/rr_ground_C.json
python grounding/rerank_ground.py --f results/runs/asqa_cands_C.json --ground results/runs/cand_ground_C.jsonl \
    --split critique/splits/asqa_C.json --mode mix --agg mean --param 0.75 \
    --rel results/runs/cand_qarel_C.jsonl --delta 1 --out out/rr_ground_v2_C.json
python grounding/lookup_perq.py --f out/rr_ground_C.json --metrics results/runs/cand_metrics_C_exact.jsonl
python grounding/lookup_perq.py --f out/rr_ground_v2_C.json --metrics results/runs/cand_metrics_C_exact.jsonl
python critique/bootstrap.py --base out/rr_released_lookup_C.json.perq.jsonl --new out/rr_ground_C.json.perq.jsonl \
    --split critique/splits/asqa_C.json --out out/rr_ground_C.bootstrap.json
python critique/bootstrap.py --base out/rr_released_lookup_C.json.perq.jsonl --new out/rr_ground_v2_C.json.perq.jsonl \
    --split critique/splits/asqa_C.json --out out/rr_ground_v2_C.bootstrap.json
```

Outputs are copied to `results/runs/` and committed immediately, before the write-up.
