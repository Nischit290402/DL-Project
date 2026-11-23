# Improving Self-RAG on long-form question answering (ASQA)

Code and run outputs for the AI6103 group project report *Diagnose, Then Re-rank: Improving Self-RAG on Long-Form
Question Answering Without Retraining*.

The project reproduces Self-RAG 7B ([paper](https://arxiv.org/abs/2310.11511)) on ASQA, measures where its long
answers lose content and support, and adds three components at inference time: a passage reranker, the paper's
full critique score, and an entailment check with a relevance gate. No model is trained.

`self-rag/` is the official code ([AkariAsai/self-rag](https://github.com/AkariAsai/self-rag), commit
`1fcdc420e48f50a7d7ab1ece5494221b93252e99`, MIT license) with the flags this project added.

## 1. Reproduce every table of the report (CPU, about 2 minutes)

Requirements: `bash` and Python 3.8 or newer. No packages, no GPU, no downloads.

```bash
git clone https://github.com/Nischit290402/DL-Project.git
cd DL-Project
bash scripts/reproduce_report.sh
```

The script recomputes each table from the run outputs committed in `results/runs/`: it picks the answers again
with the project's ranking code, checks that they are identical to the committed answer files, and recomputes
every difference and its 95% interval. Each block prints an `expected` line with the report's numbers.

| Report item | Block printed by the script | Code that computes it |
| --- | --- | --- |
| Table 1, reproduction (C0, C1) | `Table 1` | score files written by `cluster/run_alce_eval.sbatch` |
| Table 2, combined system on Set C and on all 948 | `Table 2` | `combined/run_ladder.sh`, `combined/rerank_combined.py`, `critique/bootstrap.py` |
| Tables 3 and 4, ablation over all eight combinations | `Tables 3 and 4` | `combined/factorial.py` |
| Results paragraph on the likelihood term (7, 65, 66 of 200) | `Results paragraph` | `critique/paper_scoring_check.py` |
| Table 5, ideas that did not pass | `Table 5` | `grounding/rerank_ground.py`, `contradiction/`, `critique/learn_weights.py`, `retrieval/analyze_runs.py` |
| Table 6, second segment (exploratory) | `Table 6` | `retrieval/second_segment_pick.py`, `scripts/second_segment_table.py` |

Notes on reading the output:

- Tables 3, 4 and 6 use 747 of the 748 Set C questions. One question has a two-segment answer and is not in the
  saved per-candidate scores, so these tables differ from Table 2 in the second decimal (32.87 against 32.86).
- Intervals are paired bootstrap intervals over questions with a fixed seed, so they are identical on every run.
- MAUVE is computed once over a whole answer file on the GPU and is read from the committed score files.
- The diagnosis numbers (D1 to D4) and the temperature-calibration row of Table 5 are documented with their
  commands in `results/asqa_reproduction.md`, `results/retrieval.md`, `results/critique_p1.md` and
  `results/critique_p2.md`.

A second offline check, written earlier by the team, re-runs each division's own steps:
`bash scripts/check_offline.sh` (about 1 minute).

## 2. Repository layout

```
self-rag/        official Self-RAG code. retrieval_lm/run_long_form_static.py has the project's flags
                 (--keep_special_tokens, --save_candidates, --indices_file, ...);
                 retrieval_lm/run_second_segment.py writes the second segment of Table 6
retrieval/       passage reranker: answer coverage, cross-encoder reranking, run analysis, second-segment choice
critique/        full critique score, question sets (splits/), per-question ALCE scoring, bootstrap
grounding/       entailment (NLI) support scores, relevance gate, grounded ranking
contradiction/   contradiction penalty (tested and dropped, Table 5)
combined/        the combined system: ranking with any subset of the components, ladder, ablation, test plan
cluster/         environment setup, data download and every GPU job (SLURM)
scripts/         reproduce_report.sh (section 1), check_offline.sh
results/         write-ups (*.md) and committed run outputs (results/runs/)
```

Names used in the code and write-ups: **C0** is the released code, **C1** is C0 with `--keep_special_tokens` (the
repaired baseline of the report). Sets A and B (100 questions each) are for development, Set C (748) is held
out. `D2-v2` in file names is the entailment check with the relevance gate.

## 3. Regenerate the run outputs on a GPU

Section 1 starts from saved model outputs. This section lists the jobs that produced them, in order. They were
run on one NVIDIA L40S (46 GB) under SLURM. On other hardware about 3% of answers change through ties between
near-equal candidates, so numbers can differ in the first decimal.

Setup, once (about 15 minutes plus downloads):

```bash
git clone https://github.com/Nischit290402/DL-Project.git ~/DL-Project && cd ~/DL-Project && mkdir -p logs out
bash cluster/setup_env.sh      # conda environment at ~/envs/selfrag (vllm 0.2.x, see cluster/requirements-lock.txt)
bash cluster/get_data.sh       # Self-RAG 7B (13 GB) and the ASQA file with GTR top-100 passages
bash cluster/setup_alce.sh     # ALCE scorer, including the 45 GB T5-11B citation model
python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-reranker-base')"
sbatch cluster/run_smoke.sbatch   # logs/smoke_<job>.log must end with SMOKE_OK
```

`cluster/*.sbatch` are written for NTU's TC2 cluster (partition `MGPU-TC2`). On another cluster, change the
`#SBATCH` lines and `cluster/env.sh`.

Jobs, in order. Each writes to `out/`. The committed copies are in `results/runs/`.

```bash
AB=critique/splits/asqa_A.json,critique/splits/asqa_B.json
C=critique/splits/asqa_C.json

# Table 1: C0 and C1 on all 948 questions (about 50 minutes each), then ALCE scores
sbatch --export=ALL,START=0,TAG=baseline --time=03:00:00 cluster/run_asqa.sbatch
sbatch --export=ALL,START=0,TAG=keeptok,EXTRA=--keep_special_tokens --time=03:00:00 cluster/run_asqa.sbatch
sbatch --export=ALL,F=out/asqa_baseline_0_end.json cluster/run_alce_eval.sbatch
sbatch --export=ALL,F=out/asqa_keeptok_0_end.json  cluster/run_alce_eval.sbatch

# Reranker: score the GTR passages with the cross-encoder, then run Self-RAG on the reranked top 5
sbatch cluster/run_retrieval_rerank.sbatch                       # writes data/reranked/asqa_R2_K20.json
bash retrieval/submit_run.sh rrR2K20  5 $AB data/reranked/asqa_R2_K20.json
bash retrieval/submit_run.sh rrR2K20C 5 $C  data/reranked/asqa_R2_K20.json

# Combined system: ALCE scores, entailment and relevance of every candidate (about 1 hour), then the ranking (CPU)
sbatch cluster/run_combined_prep.sbatch                          # log must end with COMBINED_PREP_OK
bash combined/run_ladder.sh                                      # Table 2 and out/combined/final_948.json
python3 combined/factorial.py                                    # Tables 3 and 4
sbatch --export=ALL,F=out/combined/final_948.json,FLAGS=--mauve cluster/run_critique_score.sbatch

# Table 6: second segment. IDX must be a single split file (sbatch splits --export values on commas)
S=results/runs/combined/second
sbatch --export=ALL,TAG=AB,IDX=$S/split_AB.json cluster/run_second_segment.sbatch
sbatch --export=ALL,TAG=BC,IDX=$S/split_BC.json cluster/run_second_segment.sbatch
sbatch --export=ALL,TAG=SAME,IDX=$S/split_ALL.json,EXTRA=--same_passage cluster/run_second_segment.sbatch
sbatch --export=ALL,TAG=AB,SPLIT=$S/split_AB.json    cluster/run_seg2_ground.sbatch
sbatch --export=ALL,TAG=BC,SPLIT=$S/split_BC.json    cluster/run_seg2_ground.sbatch
sbatch --export=ALL,TAG=SAME,SPLIT=$S/split_ALL.json cluster/run_seg2_ground.sbatch
python3 retrieval/second_segment_pick.py results/runs/combined/final_948.json out/seg2_filter_948.json filter 0.5 \
  out/seg2_AB.json,out/seg2_BC.json out/seg2_AB.ground.jsonl,out/seg2_BC.ground.jsonl \
  out/seg2_AB.ground_ev.jsonl,out/seg2_BC.ground_ev.jsonl
sbatch --export=ALL,F=out/seg2_filter_948.json,FLAGS=--mauve cluster/run_critique_score.sbatch
```

The other rows of Table 6 use the same last two commands with rule `beam` in place of `filter`, and with the
three `seg2_SAME` files. `scripts/reproduce_report.sh` shows the exact calls.

The runs behind Table 5 and the diagnosis have their commands in each division's README: `retrieval/README.md`
(oracle passages, 10 or 20 passages), `grounding/` and `results/nli_grounding_*.md` (entailment check with and
without the gate), `contradiction/README.md`, and `results/critique_p1.md`, `results/critique_p2.md`
(calibration, learned weights).

## 4. Settings

| Setting | Value |
| --- | --- |
| Generator | `selfrag/selfrag_llama2_7b`, 5 passages, beam width 2, weights 1 / 1 / 0.5, greedy decoding |
| Passages | ALCE's GTR top 100 for ASQA; the reranker reorders the top 20 with `BAAI/bge-reranker-v2-m3` and keeps 5 |
| Entailment check | `cross-encoder/nli-deberta-v3-large`, mix coefficient 0.75 |
| Relevance gate | `BAAI/bge-reranker-base`, margin 1 |
| Second segment | accepted when mean evidence-sentence entailment is at least 0.5 |
| Scoring | ALCE (commit `246c476`): STR-EM, ROUGE-L, MAUVE, citation recall and precision |

## 5. Known issues in the official code

- `run_long_form_static.py` served through vLLM drops Self-RAG's reflection tokens unless
  `--keep_special_tokens` is set. Citation recall falls from 66.6 to 50.2 without it (`results/asqa_reproduction.md`).
- The released ranking reads the utility probabilities at the support tokens and never passes the sequence
  score, so two of the four score terms have no effect (`results/critique_paper_scoring.md`).
- `run_short_form.py` passes an unused `max_depth` argument that raises a `TypeError`. Fixed here.
- The code needs the vLLM 0.2.x line, because it asks for full-vocabulary log-probabilities.
