# ASQA reproduction (Self-RAG 7B, long-form) and the reflection-token bug

**Summary.** The released long-form script reproduces the paper's ASQA correctness numbers but not its citation
numbers (citation recall 50.2 vs 67.8). Others have reported the same gap in upstream issue
[AkariAsai/self-rag#4](https://github.com/akariasai/self-rag/issues/4), which has been open since October 2023.
The cause is a vLLM default: `SamplingParams(skip_special_tokens=True)` removes Self-RAG's reflection tokens from
the decoded text, and the script's post-processing needs those tokens to attach a citation to every sentence.
Keeping them (`--keep_special_tokens`) gives citation recall 66.6 and precision 67.6, within 0.4 of the author's own
scores, with str-em, ROUGE-L and MAUVE unchanged. The same bug also disables every text-based decision in the
script (segment expansion, the adaptive-retrieval threshold and `--ignore_cont`). Even with the fix, Self-RAG
7B never asks for a second retrieval on ASQA, so segment-level beam search stays at depth 1.

## Setup

- Cluster: NTU CCDS TC2, one NVIDIA L40S 46 GB per job. Env: `cluster/requirements-lock.txt` (vllm 0.2.6,
  torch 2.1.2+cu121, transformers 4.36.2). Model: `selfrag/selfrag_llama2_7b`.
- Data: `data/eval_data/asqa_eval_gtr_top100.json` (ALCE ASQA dev, 948 questions, GTR top-100 passages).
- Inference: `cluster/run_asqa.sbatch` with the upstream README settings: `--mode always_retrieve --ndocs 5
  --max_new_tokens 300 --threshold 0.2 --use_grounding --use_utility --use_seqscore --max_depth 7
  --beam_width 2 --dtype half`, critique weights 1.0 / 1.0 / 0.5 (the values upstream hardcodes).
- Scoring: ALCE `eval.py` at commit `246c476` with `--citations --qa --mauve` (`cluster/run_alce_eval.sbatch`).
  Citation metrics use `google/t5_xxl_true_nli_mixture`.

| Run | Job | Flag | Wall time |
| --- | --- | --- | --- |
| Baseline, released code | 52720 (scored by 52771) | none | 53.3 min |
| Fixed | 52783 (scored by 52809) | `--keep_special_tokens` | 40.2 min |
| Pilots, first 30 questions | 52656, 52774 | none / `--keep_special_tokens` | 1.8 / 1.3 min |

Raw predictions: `out/asqa_baseline_0_end.json`, `out/asqa_keeptok_0_end.json` (+ `.score`) on the cluster,
not committed.

## Results (948 questions)

| Run | str-em | ROUGE-Lsum | MAUVE | Citation rec | Citation prec | QA-F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Released code (tokens stripped) | 30.13 | 35.83 | 76.09 | 50.22 | 63.92 | 24.02 |
| **`--keep_special_tokens`** | **30.19** | **35.75** | **75.57** | **66.64** | **67.57** | 24.10 |
| Author's own scores, issue #4 | 29.96 | 35.70 | 74.33 | 66.97 | 67.82 | - |
| Independent rerun, issue #4 (Jul 2024) | 30.23 | 35.76 | 76.23 | 50.19 | 63.92 | - |
| Paper, Table 2 | 30.0 | 35.7 | 74.3 | 67.8 | 66.9 | - |

- Our released-code run matches the independent rerun in issue #4 to within 0.1 on every metric.
- With the fix, citation recall rises by 16.4 points and precision by 3.7 points; correctness and fluency move by
  0.5 or less. The fixed run lands within 0.4 of the author's own scores.
- The paper lists precision 66.9 / recall 67.8; the author's ALCE output has recall 66.97 / precision 67.82, so
  the two columns in Table 2 look swapped. Check against the PDF before quoting this.

Pilot (first 30 questions, `--citations --qa` only): citation recall 49.2 → 71.7, precision 70.0 → 71.7,
str-em 31.9 → 30.8.

## Root cause

vLLM's `SamplingParams` defaults to `skip_special_tokens=True`. Self-RAG adds its reflection tokens
(`[Retrieval]`, `[No Retrieval]`, `[Continue to Use Evidence]`, `[Relevant]`, `[Fully supported]`,
`[Utility:1-5]`, ...) as `additional_special_tokens` (`retrieval_lm/finetune.py`), so they never reach the
decoded text. `run_long_form_static.py` scores candidates from token ids and log-probs, which is unaffected, but
several decisions read the decoded text:

| Code (upstream `run_long_form_static.py`) | Intended | With tokens stripped |
| --- | --- | --- |
| `if "[Retrieval]" in pred:` in `call_model_beam_batch` | expand a beam node with a new segment | never true, so beam search never passes depth 1 |
| `if "[No Retrieval]" in pred_text:` in `run_step_generation_batch` | apply `--threshold` to turn `[No Retrieval]` into a retrieval | never true, so the threshold has no effect |
| `"[No support / Contradictory]" not in processed_pred` | `--ignore_cont` drops unsupported segments | never matches, so the flag does nothing |
| `.replace(".[Continue to Use Evidence]", " [1]. ")` | give each sentence of a segment its own citation | never matches, so only the last sentence is cited |

The last row is what moves the scores. A typical generation is

```
[Relevant]The song "The Sound of Silence" was originally performed by ... Simon & Garfunkel.[Fully supported]
[Continue to Use Evidence]The song was written by Paul Simon over ... 1963 and 1964.[Utility:5]</s>
```

With the tokens kept, the output is `S1 [1]. S2 [1].`; with them stripped it is `S1. S2 [1].` ALCE's citation
recall scores a sentence without a citation as unsupported, so every non-final sentence of a multi-sentence
answer counts against the released code.

Why the comparison is controlled: decoding is greedy, the prompts are the same, and candidate scores come from
token ids, so both runs should generate the same tokens and pick the same candidates. Only the decoded text and
the citation markers should differ. `cluster/compare_outputs.py` checks this (see Verification).

PopQA (`run_short_form.py`) is not affected in the same way: its retrieval decision and scores use log-probs and
token ids, and the answer text is cleaned anyway. This fits the PopQA baseline matching the paper
(`results/popqa_baseline.md`).

## Segment-level beam search is inert on ASQA

Counts in the 948 baseline outputs: `[Retrieval]` 948 (only the root placeholder the code writes once per
question), `[No Retrieval]` 0, `[Continue to Use Evidence]` 0, because the tokens are stripped. In the fixed
30-question pilot the tokens are present (`[Relevant]` 30, `[Utility:` 30, `[Fully supported]` 27,
`[Continue to Use Evidence]` 17), but `[Retrieval]` and `[No Retrieval]` are still never generated after the
first segment. The model writes the whole answer in one segment, staying on the first passage with
`[Continue to Use Evidence]`. So with or without the fix, every answer is one segment from one passage. The only
ranking that matters is the first step: 5 candidates (one per passage), keep 2, output the best.

Consequences for our experiments:
- On ASQA, anything that acts on later beam steps or compares segments (self-contradiction across segments,
  NLI-based ranking of later segments) has nothing to act on unless the retrieval trigger is changed.
- Every ASQA experiment should be run with `--keep_special_tokens` and compared with the fixed run (66.64 / 67.57),
  not with the released-code run.

## Verification

- [ ] `grep -o "\[Retrieval\]" out/asqa_keeptok_0_end.json | wc -l` gives 948 (no segment expansion in the full fixed run).
- [ ] `cluster/compare_outputs.py out/asqa_baseline_0_end.json out/asqa_keeptok_0_end.json` shows identical
      generations for all 948 questions and more citation markers per answer in the fixed run.
- [ ] Table 2 column order checked against the paper PDF.

## Code changes (branch `asqa-baseline`)

- `self-rag/retrieval_lm/run_long_form_static.py`: `--keep_special_tokens` (passes `skip_special_tokens=False` to
  every `SamplingParams`); `--start/--end`; critique weights read from `--w_rel/--w_sup/--w_use` (upstream
  hardcoded 1.0/1.0/0.5, so the `--w_use` default is now 0.5); `--gpu_memory_utilization`; progress timing.
  Defaults reproduce upstream behaviour.
- `cluster/`: `run_asqa.sbatch`, `setup_alce.sh`, `env_alce.sh`, `run_alce_eval.sbatch`, `contradiction_rate.py`,
  `run_contra.sbatch`, `merge_shards.py`, `compare_outputs.py`.
