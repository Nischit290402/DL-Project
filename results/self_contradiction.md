# Self-contradiction: useless for re-ranking, confirmed as an uncertainty signal

**Summary.** The original idea, penalising contradictions between segments of one long-form answer, cannot be
tested on ASQA: Self-RAG writes 947 of 948 answers as a single segment. We moved the idea to the 5 candidate answers
Self-RAG writes per question (one per retrieved passage) and penalised candidates that the others contradict,
judged by an NLI model. Diagnostics on Set A showed a real but weak signal, weaker than Self-RAG's own ranking, and
a hand check found only 11 of 30 flagged pairs to be genuine contradictions. With the setting fixed on Set A
(lambda 0.3), the held-out test on Set B gives **no gain in correctness (str-em -0.75 [-3.92, +2.08]) and a
significant loss in citations (recall and precision -7.53, intervals [-14.0, -2.2] and [-13.5, -1.7])**. The
penalty overrides Self-RAG's support grade: on Set B the answers it swaps in are judged supported by their own
passage 62% of the time, against 95% for the answers they replace. Both findings were then tested on the 748
unused questions (Set C) with the tests written down beforehand (`contradiction/PREREG_C.md`). The penalty again
gives no str-em gain (+0.17 [-0.84, +1.13]); its citation loss is smaller than on Set B and just misses the
pre-registered rule (-2.24 [-4.17, +0.01], not confirmed). **Disagreement among candidates is confirmed as an
uncertainty signal: in the less contradictory half of Set C the released answer scores +8.36 [+3.98, +13.23]
str-em and +12.77 [+5.80, +18.56] citation recall higher than in the more contradictory half.**

Code: `contradiction/` and `cluster/run_cand_nli.sbatch`, `cluster/run_contra_eval.sbatch` (branch
`self-contradiction`, from `critique-p1`). All GPU runs on NTU TC2, NVIDIA L40S.

## 1. Why the original idea cannot be tested on ASQA

The idea: during segment-level beam search, check each new segment against the earlier ones with an NLI model and
penalise contradictions. On ASQA there are almost no later segments to check. C0 (released code) has no answer with
two or more segments and C1 (`--keep_special_tokens`) has one in 948 (1 segment pair in total,
`results/critique_p1.md`, Result 1). The model writes the whole answer after the first retrieval and continues with
`[Continue to Use Evidence]` instead of asking for another passage (`results/asqa_reproduction.md`). Creating
segments would need a new retrieval trigger, which changes the system being compared, so we did not pursue it.

## 2. Reformulated method

For each question Self-RAG writes 5 candidates, one per passage, and keeps the one with the highest score
(relevance + support in practice; utility and the sequence score are inactive, `critique/HANDOFF_P2.md`). It never
compares the candidates with each other. Hypothesis: a candidate that the others contradict is more likely wrong.

- NLI model: `cross-encoder/nli-deberta-v3-large`, every ordered pair of candidates of a question, two variants:
  plain (`noctx`) and with "Question: ... Answer:" in front of the premise (`ctx`). Not the T5-11B model that ALCE
  uses to grade citations, so the ranking is not tuned to the grader.
- C_i = mean over the other 4 candidates of max(p_con(i->j), p_con(j->i)).
- New score_i = released score_i - lambda * C_i. Answers re-picked offline from the saved candidates
  (`results/runs/asqa_cands_0_end.json`, Set A + B); lambda 0 reproduces C1's answers on all 200 questions.
- Protocol: diagnostics and tuning on Set A only; one test on Set B with the setting fixed beforehand (commit
  `bb39fcf`, before any Set B output existed). Scored with `critique/score_per_question.py` and compared with the
  95% paired bootstrap of `critique/bootstrap.py` against `results/runs/rr_released.json` (= C1's answers).

## 3. Diagnostics on Set A (go/no-go)

Full numbers: `results/cand_contra_A.json`.

**D1, how often candidates contradict each other.** 88% of questions (89% with question context) have at least one
pair with p_con >= 0.5, and 56% of all candidate pairs are flagged. Raising the threshold to 0.9 changes this by
about one point, so the NLI model is close to binary on these inputs.

**D2, does C predict errors.** Within each question, among candidate pairs with different str-em, the share where
the better candidate has the lower score (0.5 = no signal):

| Signal | noctx | ctx |
| --- | --- | --- |
| Contradiction C | 0.586 [0.503, 0.669] | 0.607 [0.514, 0.688] |
| Entailment agreement | 0.618 [0.535, 0.700] | 0.597 [0.507, 0.687] |
| Word-overlap agreement | 0.588 [0.504, 0.670] | same |
| Self-RAG's released score | 0.682 [0.609, 0.753] | same |

str-em by C (noctx): 33.7 (C < 0.1, n 94), 28.3 (0.1-0.3, n 63), 20.1 (0.3-0.5, n 37), 17.5 (>= 0.5, n 306).
The signal passes the stop rule narrowly but is weaker than Self-RAG's own ranking and no better than word overlap.
In the 19-22 questions where the chosen answer is the most contradicted candidate, the least contradicted
alternative has lower str-em (noctx 17.5 vs 20.4, ctx 22.9 vs 26.2): where the two signals disagree most,
Self-RAG is right more often.

**D3, what the flags are.** 30 flagged pairs (ctx, p_con >= 0.5) drawn at random and labelled
(`results/cand_contra_A_labels.csv`):

| Label | Pairs | Example |
| --- | --- | --- |
| genuine: cannot both be true for one reading of the question | 11 | "Army's last bowl game: Dec 2010" vs "Dec 1984" |
| different-reading: both can be true, they answer different readings | 13 | "Phillies won in 1980" vs "won in 2008" (both correct) |
| nli-error: no real conflict | 6 | "average cigarettes in a pack 16.2" vs "average smoked per day 15.8" |

In 6 of the 11 genuine pairs both answers are wrong, so a contradiction marks uncertainty rather than pointing to
the right candidate. Of the 4 sampled pairs flagged only with question context, 3 are NLI errors.

## 4. Setting fixed on Set A

str-em on Set A for new score = released - lambda * C (released 27.78):

| Variant | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 | 0.75 | 1 | 1.5 | 2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| noctx | 27.62 | 27.62 | 28.30 | **28.50** | 28.25 | 27.75 | 27.42 | 26.75 | 26.42 |
| ctx | 27.22 | 27.72 | 27.67 | 28.07 | 28.72 | 27.38 | 27.38 | 26.72 | 24.72 |

An agreement bonus (+ mu * entailment) never exceeded 28.22. Chosen: **noctx, lambda 0.3**, the middle of the
0.2-0.5 plateau (the ctx peak is a single point, and D3 shows context adds false alarms). On Set A this is
+0.72 [-2.00, +3.25] str-em, 21 of 100 answers changed (6 better, 4 worse): not significant even on the tuning set.

## 5. Held-out test on Set B (100 questions)

| Metric | Released (C1) | Penalty | Penalty - released [95% CI] |
| --- | --- | --- | --- |
| str-em | 33.92 | 33.17 | -0.75 [-3.92, +2.08] |
| ROUGE-Lsum | 37.62 | 36.76 | -0.87 [-2.65, +0.66] |
| Citation recall | 70.35 | 62.82 | **-7.53 [-14.00, -2.20]** * |
| Citation precision | 71.67 | 64.13 | **-7.53 [-13.53, -1.67]** * |
| Length (words) | 27.21 | 29.42 | +2.21 [-0.06, +5.00] |

`*` = interval excludes 0. 21 of 100 answers change; the whole citation loss comes from them (on those 21, citation
recall 92.9 -> 57.0, str-em 36.1 -> 32.5).

## 6. Why the citations drop

Changed answers only, original (Self-RAG's choice) vs swapped in by the penalty:

| | Set A (21) original | swapped in | Set B (21) original | swapped in |
| --- | --- | --- | --- | --- |
| C (contradiction from the others) | 0.825 | 0.470 | 0.621 | 0.298 |
| Self-RAG support grade | 0.963 | 0.926 | 0.922 | 0.882 |
| Outside checker: supported by own passage | 76% | 71% | **95%** | **62%** |
| Content words found in own passage | 0.77 | 0.76 | 0.81 | 0.79 |
| Content words found in the other passages | 0.69 | 0.75 | 0.71 | 0.75 |
| Words | 31.4 | 29.6 | 26.5 | 36.9 |

"Outside checker" = the T5-11B verdict per candidate from `results/runs/asqa_cands_0_end.json.verdicts.jsonl`
(`critique/nli_verdicts.py`). Content-word overlaps are lexical, stop words removed.

The penalty buys agreement with the other candidates by giving up support: the answers it swaps in carry a lower
support grade and, on Set B, are much less often supported by the passage they cite. Candidates written from
different passages naturally state different facts, so a well-grounded candidate is often the one that disagrees.
The swapped-in answers share slightly more content with the other passages, which fits the idea that agreement
partly reflects what the model says regardless of its passage, but that evidence is weak (lexical, 21 answers).

## 7. Contradiction as an uncertainty signal (exploratory on Sets A and B; confirmed on Set C, section 9)

Question-level disagreement = mean p_con over the 10 candidate pairs. Split each set at its median:

| Set | Chosen str-em, low disagreement | high disagreement | AUROC (disagreement -> chosen str-em = 0) |
| --- | --- | --- | --- |
| A | 32.83 | 22.73 | 0.618 |
| B | 38.53 | 29.30 | 0.522 |

Disagreement among candidates goes with lower correctness on both sets (about 10 points), but it is a weak detector
of fully wrong answers. Using it to flag uncertain answers, rather than to re-rank, is the more promising use.

## 8. Limitations

- Only one NLI model and one form of penalty were tested; the Set B estimate of the citation loss (-7.5) was not
  borne out at that size on Set C (-2.2), a reminder that 100-question estimates are noisy.
- ASQA rewards covering several readings of a question, which makes "contradiction" partly the wrong target here;
  a dataset with single-answer questions may behave differently.
- The D3 labels are one annotator's judgement on 30 pairs.

## 9. Held-out confirmation on Set C (pre-registered)

Set C = the 748 ASQA questions in neither Set A nor Set B. The tests, decision rules and settings were committed in
`contradiction/PREREG_C.md` (commit `f94597b`) before the Set C job ran; one job produced everything
(`cluster/run_contra_C.sbatch`, job log `results/runs/contra_C_log.txt`, L40S, about 1 h).

**Sanity check.** Released choices re-picked from the new Set C candidates match C1 exactly on all 748 questions
(difference 0.00 on every metric), so Set C is directly comparable with the reference run.

**H1 (primary): disagreement flags less correct answers.** Split at the median disagreement (0.600):

| Released answer | Low disagreement (374) | High disagreement (374) | Low - high [95% CI] |
| --- | --- | --- | --- |
| str-em | 34.19 | 25.84 | **+8.36 [+3.98, +13.23]** |
| Citation recall | 72.32 | 59.55 | +12.77 [+5.80, +18.56] |
| ROUGE-Lsum | 36.85 | 34.93 | +1.92 [-0.41, +4.01] |

**Confirmed** (interval above 0). Sets A + B had suggested +9.67. As a detector of fully wrong answers (str-em 0,
311 of 748) the signal is weak but above chance: AUROC 0.594 [0.553, 0.636].

What this would buy in practice (descriptive, not pre-registered): answering only the least contradictory questions
and flagging the rest.

| Questions answered | 25% | 50% | 75% | 100% |
| --- | --- | --- | --- | --- |
| str-em | 36.16 | 34.19 | 32.95 | 30.02 |
| Citation recall | 71.35 | 72.32 | 70.00 | 65.93 |

**H2: the penalty (noctx, lambda 0.3) vs released choices.** 158 of 748 answers change.

| Metric | Released | Penalty | Difference [95% CI] | Pre-registered rule | Verdict |
| --- | --- | --- | --- | --- | --- |
| str-em | 30.02 | 30.19 | +0.17 [-0.84, +1.13] | H2a: interval not above 0 | **confirmed: no gain** |
| Citation recall | 65.93 | 63.69 | -2.24 [-4.17, +0.01] | H2b: interval below 0 | **not confirmed** |
| Citation precision | 66.94 | 64.88 | -2.05 [-3.98, +0.16] | reported only | |
| ROUGE-Lsum | 35.89 | 35.78 | -0.10 [-0.60, +0.34] | reported only | |

On the 158 changed answers, citation recall falls from 62.3 to 51.7 and precision from 63.5 to 53.8, while str-em
moves from 27.5 to 28.3. The direction matches Set B and section 6, but over all 748 questions the loss is about a
third of the Set B estimate and its interval reaches 0.

**Conclusion.** Contradiction between candidates does not help choose a better answer, but how much the candidates
contradict each other is a reliable, held-out sign of how far to trust the answer Self-RAG gives.

## Files

| File | Content |
| --- | --- |
| `results/runs/cand_nli_A.jsonl`, `cand_nli_B.jsonl` | NLI probabilities for every ordered candidate pair, both variants |
| `results/cand_contra_A.json` | D1, D2 |
| `results/cand_contra_A_pairs.csv`, `results/cand_contra_A_labels.csv` | D3: all flagged pairs; 30 labelled |
| `results/runs/rr_contra_B.json` (+ `.perq.jsonl`, `.summary.json`) | Set B answers with the penalty, scores per question |
| `critique/splits/asqa_C.json`, `contradiction/PREREG_C.md` | Set C and the tests fixed before it was run |
| `results/runs/asqa_cands_C.json`, `cand_nli_C.jsonl` | Set C candidates and NLI scores |
| `results/runs/rr_released_C.json`, `rr_contra_C.json` (+ `.perq.jsonl`, `.summary.json`) | Set C answers and scores |
| `results/uncertainty_C.json`, `results/contra_C_bootstrap.json`, `results/runs/contra_C_log.txt` | H1, H2, job log |
