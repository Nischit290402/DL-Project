# NLI grounding (D2), Step 4: Set B test of the fixed setting

Setting fixed on Set A and committed before any Set B run: commit `e740080`, `grounding/configs/d2_final.json`
(mode `mix`, aggregation `mean`, α = 0.75). Set B was run once with it; nothing was changed afterwards.

## Result (Set B, 100 questions, paired 95% bootstrap vs released = C1)

| Metric | Released | D2 | D2 − released |
| --- | --- | --- | --- |
| str-em | 33.92 | 33.68 | −0.23 [−5.47, +4.37] |
| ROUGE-Lsum | 37.62 | 34.74 | **−2.88 [−5.41, −0.52] \*** |
| Citation recall | 70.35 | 76.33 | +5.98 [−0.75, +12.88] |
| Citation precision | 71.67 | 77.50 | +5.83 [−0.58, +12.83] |
| Length (words) | 27.21 | 28.10 | +0.89 [−3.00, +5.29] |

\* interval excludes 0. 45 picks changed, 40 with different answer text.

**Verdict on Set B: citation goes up as on Set A (+6.0 vs +9.8) but the interval just includes 0; str-em is flat;
ROUGE-L drops significantly.** H-D2 ("citation up without hurting answer quality") is not confirmed on Set B.

Files: `results/runs/rr_ground_B.json` (+ `.perq.jsonl`, `.config.json`). Scores were computed by lookup in the
exact per-candidate table (`grounding/lookup_perq.py`), which reproduces `score_per_question.py` exactly: the
zero setting gives the released per-question scores on 200/200 questions, every metric.

```bash
python grounding/rerank_ground.py --f results/runs/asqa_cands_0_end.json --ground results/runs/cand_ground_B.jsonl \
    --split critique/splits/asqa_B.json --mode mix --agg mean --param 0.75 --out out/rr_ground_B.json
python grounding/lookup_perq.py --f out/rr_ground_B.json --metrics results/runs/cand_metrics_AB_exact.jsonl
python critique/bootstrap.py --base results/runs/rr_released.json.perq.jsonl \
    --new results/runs/rr_ground_B.json.perq.jsonl --split critique/splits/asqa_B.json
```

## What the swapped answers look like (40 questions with a different answer)

| On the 40 changed questions | Released answer | D2 answer |
| --- | --- | --- |
| Citation recall | 62.1 | **77.1** |
| Fully supported (citation 100) | 23 | 26 |
| str-em | 38.8 | 38.2 |
| ROUGE-Lsum | 40.6 | **33.4** |
| Sentences | 1.57 | 1.62 |

Per question: citation better on 15 / worse on 7; str-em better 9 / worse 10; **ROUGE better 13 / worse 25**.

This is the reverse of sy's D4 failure: D4 swapped in poorly grounded answers (62% supported vs 95%); D2 swaps in
better-grounded ones, so citation rises as intended.

**The ROUGE loss has one clear cause: the grounding score asks "is this supported by its own passage?", not
"does this answer the question?".** The largest ROUGE drops are faithful-to-passage, off-question answers:

- *Where does the Blue Lagoon water come from?* Released: the Icelandic geothermal spa. D2: a Blue Lagoon in
  Pembrokeshire, Wales (fully supported by its passage, wrong entity).
- *When did Ben Rector's new album come out?* Released: 2013 and 2015 albums. D2: a 2010 album.
- *How many teams can play in the Champions League?* Released: 32 teams in the group stage. D2: 76 teams entered
  the 2012-13 qualifying (supported, but not what the question asks).
- *Vikram Samvat is the official calendar of which country?* Released states Nepal; D2 describes the Nepali New
  Year instead.

The relevance term (`[Relevant]` ≈ 1 for almost every candidate) does not stop this, so grounding decides near-ties
between candidates that are equally "relevant" by Self-RAG's own grade. ROUGE (overlap with the reference long
answers) is the metric that notices the topic drift.

On Set A the same direction was visible but not significant (ROUGE −1.15 [−2.9, +0.5]): the tuning rule only
constrained str-em, so it did not guard against this.

## Options for Set C (team decision; nothing below has been run)

1. **Pre-register the fixed setting as is** (protocol-clean): primary = citation recall gain on Set C; secondary
   = str-em and ROUGE-L non-inferiority. Set C (748 questions) has about 2.7× narrower intervals than Set B, so a
   +6 citation effect would be detectable; the likely outcome is "citation up, ROUGE down: a trade-off".
2. **Fix the failure mode, then go straight to Set C.** A question-aware variant, for example grounding only
   counts for candidates that also answer the question (premise "Question: q" + passage, or a gate on an
   answer-relevance score), tuned on Set A only with a ROUGE constraint added to the rule. Set B is then no longer
   an untouched test for that variant (it motivated it) and must be reported as such; Set C is the only clean
   test. sy found the "Question:" premise variant raised false positives, so this needs its own Set A diagnostic.
3. **Both**, with option 1 as the pre-registered primary test and option 2 as a clearly labelled exploratory row.

Recommendation: option 3. It keeps the confirmatory test honest and still reports the obvious improvement.
