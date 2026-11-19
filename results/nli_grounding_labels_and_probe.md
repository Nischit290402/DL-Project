# NLI grounding (D2): hand-labelled error analysis on A+B, and why DeBERTa misses verbatim support

Date 2026-11-19. **Labels: every label (90 swaps, 30 G4 sentences) was checked and confirmed by hand by Boxuan Li**
in the labelling tool (`results/label_tool/index.html`). Draft labels were pre-filled with an LLM assistant to
speed up the pass; the numbers below are the human-confirmed labels. Source: `results/label_tool/claude_labels.json`
(file name kept from the draft step);
CSVs: `results/label_tool/labels_rouge_swaps_AB.csv`, `results/label_tool/labels_G4.csv`.
Build: `python grounding/make_label_tool.py`.

## 1. The 90 answer swaps on Set A+B (D2 vs released, different answer text)

97 picks changed (A 52, B 45); 90 change the answer text (A 50, B 40). Every one was labelled with a main reason and
"which answer is better" after reading the question, its disambiguated short answers, the reference long answers,
both answers and their passages.

| Main reason | All 90 | ROUGE down (51) | ROUGE flat/up (39) |
| --- | --- | --- | --- |
| D2 fixes an error / covers more | 19 | 3 | 16 |
| other reading (often not in refs) | 14 | 10 | 4 |
| ROUGE misjudges (D2 as good) | 14 | 14 | 0 |
| equivalent | 13 | 2 | 11 |
| non-answer (same topic, does not answer) | 9 | 6 | 3 |
| factual error | 9 | 5 | 4 |
| wrong entity | 8 | 7 | 1 |
| less coverage | 4 | 4 | 0 |

Better answer: released 33, tie 33, D2 24 (Set A 19 / 19 / 12, Set B 14 / 14 / 12).

- ROUGE's direction agrees with the judgement on 42 of 57 non-tied swaps (released better & ROUGE down 26;
  D2 better & ROUGE up 16) and disagrees on 15 (D2 better but ROUGE down 8; released better but ROUGE up 7).
- **D2's failures are mostly off-question, not unfaithful:** wrong entity + non-answer + other reading = 31/90.
  D2 picks a passage about a namesake or neighbour (Blue Lagoon in Wales, King of Spain, Port Adelaide, Whirlpool
  Galaxy, "second sound" in physics) and grounds well in it.
- **D2's wins are mostly corrections of released hallucinations** (Kobe Bryant's "100 points", Napoleon III in
  1830, Eddie Holman, a 1979 Rams Super Bowl win): 19/90.
- Implication for a better configuration: keep grounding, but stop it from moving to a passage that does not
  answer the question, e.g. gate the grounding term by question–passage relevance (retrieval rank or a QA
  relevance cross-encoder), and add a ROUGE non-inferiority term to the Set A tuning rule.

## 2. G4 (30 sentences where ISSUP and DeBERTa disagree most)

Human judgement: supported 10, partially supported 12, not supported 8. Under a strict reading DeBERTa
is right on 19 and ISSUP on 11. ISSUP misses wrong facts (11) and unsupported exact dates (7). DeBERTa misses
near-verbatim support (5), bridging or paraphrase (4), and has one false contradiction, one false positive
from general knowledge and one borderline case.

## 3. Why DeBERTa misses near-verbatim support (TC2 job 55962)

`grounding/verbatim_check.py` (CPU): of 1,489 A+B answer sentences, 327 are near-verbatim copies of their own
passage (≥ 90% as one contiguous token run) and 307 are exact copies. **110 near-verbatim (34%) and 104 exact
copies get p_ent < 0.5**, mostly as neutral (85) and sometimes as contradiction (25).

`grounding/nli_probe.py` (GPU, `cluster/run_nli_probe.sbatch`) re-scored the 110 failures and 60 passing
near-verbatim controls, changing one thing at a time:

| Variant | Failures: share p_ent ≥ 0.5 | Controls: share p_ent ≥ 0.5 |
| --- | --- | --- |
| production (slow tokenizer, "Title: t\ntext") | 0.00 (reproduces stored scores exactly) | 1.00 |
| fast tokenizer | 0.00 | 1.00 |
| no "Title:" line | 0.15 | 0.95 |
| matching passage sentence(s) ± 1 sentence | 0.74 | 0.85 |
| **matching passage sentence(s) only** | **0.93** | 0.97 |

- Not a tokenizer bug: slow and fast tokenizers give identical ids on every input. (The cached `tokenizer.json`
  cannot be read by the pinned tokenizers 0.15, which is why the code uses the slow one; that is harmless.)
- Not truncation: no input reaches 512 tokens (median 160, max 225), and failures and controls have the same
  length (161 vs 159).
- **The cause is the premise granularity.** The cross-encoder is a sentence-level NLI model. Given a whole
  ~150-word passage it often answers "neutral" (or even "contradiction") for a sentence copied from it. Given
  only the sentence(s) that contain the claim, it entails 93% of them.
- The 8 that still fail are mostly fragments from sentence splitting on decimals ("6% of the Australian resident
  population…" from "28.6%"), so the hypothesis itself is broken. That comes from the ALCE-style splitting.

Fix to test (not run yet): score each answer sentence against every passage sentence (or 1–2-sentence windows)
and take the max p_ent (SummaC-style), instead of the whole passage at once. "only_match" above uses the known
matching sentence, so it is an upper bound. The deployable max-over-sentences version needs a GPU re-run of
`cand_ground.py` on A/B/C and re-tuning on Set A only.

## 4. Feasibility of a relevance gate (TC2 job 55973)

`grounding/qa_relevance.py` scored (question, passage) for all 1,000 A+B candidates with two off-the-shelf
rerankers. Signal = relevance(D2's passage) − relevance(released passage) on the 90 swaps.

| Reranker | AUC D2-better vs released-better swaps (hand labels) |
| --- | --- |
| cross-encoder/ms-marco-MiniLM-L-12-v2 | 0.60 |
| **BAAI/bge-reranker-base** | **0.76** |
| (for reference: retrieval rank, Self-RAG relevance) | no clear separation |

Gate tried: *D2 may switch only if bge relevance of its new passage ≥ released passage's − δ*; a blocked swap falls
back to the released answer. Effects on all 200 A+B questions vs released (exact ALCE metrics from the candidate
table; the no-gate row reproduces the reported Set A and Set B numbers):

| Gate | Swaps kept | ΔROUGE-L | Δstr-em | Δcitation rec |
| --- | --- | --- | --- | --- |
| none (D2 as committed) | 90 | −2.02 | −0.12 | +7.91 |
| δ = 0 | 38 | **+0.14** | **+2.38** | +6.29 |
| δ = 1 | 50 | −0.36 | +3.17 | +5.99 |
| δ = 2 | 61 | −0.76 | +2.67 | +7.24 |

Per set at δ = 1: A ROUGE −0.11, str-em +4.17, citation +7.50; B ROUGE −0.60, str-em +2.17, citation +4.48.

**Verdict: go.** The gate removes most of the ROUGE loss, lifts str-em and keeps about 80% of the citation
gain. These numbers are point estimates without intervals; δ was looked at on A+B, so A+B is the development set
and Set C is the only clean test. The gate uses no labels; the labels only describe which swaps it blocks.
