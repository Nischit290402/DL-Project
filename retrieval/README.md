# Retrieval division

Question: on ASQA, does the choice of passages limit Self-RAG, and can a better choice raise the end metric?
Results: `results/retrieval.md`. Plan and decision rules of the reranking test: `retrieval/PREREG.md`.

Branch `retrieval` (from `critique-p1`). `self-rag/` is unchanged on this branch: every run goes through the
team's `cluster/run_asqa.sbatch` with C1 settings and differs only in the passages file or `--ndocs`.
CPU scripts use the Python standard library only.

## Files

| File | What it does | Where |
| --- | --- | --- |
| `retrieval/coverage.py` | Answer coverage of the passages at k = 1..100 (all together, best single one); checks against ALCE's own labels | CPU |
| `retrieval/analyze_runs.py` | Compares runs with a C1 reference on the same questions: str-em, best candidate, coverage of the passage picked, length; paired bootstrap | CPU |
| `retrieval/stitch_candidates.py` | Offline bound: joins the top-k saved candidates of C1 into one answer | CPU |
| `retrieval/rerank_score.py` | Cross-encoder score for every (question, passage) pair | GPU |
| `retrieval/build_reranked.py` | Input file with the GTR top-K reordered by reranker score | CPU |
| `retrieval/cov_compare.py` | Component metrics of the 5 passages given (mean@5, rank1, best@5, union@5) against GTR, paired | CPU |
| `retrieval/explore_pick.py` | Exploratory check: Self-RAG's pick vs the candidate from the first-ranked passage | CPU |
| `retrieval/submit_run.sh` | Submits one Self-RAG run (passages file, `--ndocs`, split) | login node |
| `cluster/run_retrieval_rerank.sbatch` | Scores both rerankers and builds the four input files | GPU |
| `retrieval/records/` | Decision records written on the cluster between steps, with checksums | |

## Order of the work

All commands from the repo root on the cluster, after `bash cluster/setup_env.sh` and `bash cluster/get_data.sh`.
The ALCE data (oracle passages) comes from `cluster/setup_alce.sh` or
`https://huggingface.co/datasets/princeton-nlp/ALCE-data` (`asqa_eval_gtr_top100_reranked_oracle.json`).

```bash
AB=critique/splits/asqa_A.json,critique/splits/asqa_B.json
GTR=data/eval_data/asqa_eval_gtr_top100.json
ORACLE=data/ALCE-data/asqa_eval_gtr_top100_reranked_oracle.json

# 1. What the passages hold (validation against ALCE Table 12)
python3 retrieval/coverage.py $GTR    results/runs/retrieval/coverage_gtr_perq.jsonl    critique/splits/asqa_A.json critique/splits/asqa_B.json
python3 retrieval/coverage.py $ORACLE results/runs/retrieval/coverage_oracle_perq.jsonl critique/splits/asqa_A.json critique/splits/asqa_B.json

# 2. Controls on Set A+B: C1 rerun, oracle passages, wider pool
bash retrieval/submit_run.sh ctrl5   5  $AB
bash retrieval/submit_run.sh oracle5 5  $AB $ORACLE
bash retrieval/submit_run.sh nd10    10 $AB
bash retrieval/submit_run.sh nd20    20 $AB
python3 retrieval/analyze_runs.py results/runs/asqa_cands_0_end.json out/asqa_oracle5_0_end.json out/asqa_nd10_0_end.json out/asqa_nd20_0_end.json

# 3. Offline bound for several passages per answer
python3 retrieval/stitch_candidates.py results/runs/asqa_cands_0_end.json out/asqa_stitch

# 4. Reranking test (retrieval/PREREG.md)
sbatch cluster/run_retrieval_rerank.sbatch
python3 retrieval/cov_compare.py $GTR R1-20=data/reranked/asqa_R1_K20.json R1-100=data/reranked/asqa_R1_K100.json \
        R2-20=data/reranked/asqa_R2_K20.json R2-100=data/reranked/asqa_R2_K100.json --splits critique/splits/asqa_A.json
bash retrieval/submit_run.sh rrR2K20  5 $AB data/reranked/asqa_R2_K20.json
python3 retrieval/analyze_runs.py results/runs/asqa_cands_0_end.json out/asqa_rrR2K20_0_end.json
bash retrieval/submit_run.sh rrR2K20C 5 critique/splits/asqa_C.json data/reranked/asqa_R2_K20.json
python3 retrieval/analyze_runs.py results/runs/asqa_cands_C.json out/asqa_rrR2K20C_0_end.json
```

`critique/splits/asqa_C.json` and `results/runs/asqa_cands_C.json` are on the `self-contradiction` and
`nli-grounding` branches.

## Outputs kept in git (`results/runs/retrieval/`)

| File | Run |
| --- | --- |
| `asqa_oracle5_0_end.json`, `asqa_nd10_0_end.json`, `asqa_nd20_0_end.json` | controls, Set A+B, jobs 55484, 55487, 55491 |
| `asqa_rrR2K20_0_end.json` | reranked, Set A+B, job 56495 |
| `asqa_rrR2K20C_0_end.json` | reranked, Set C, job 56496 |
| `scores_R1.jsonl`, `scores_R2.jsonl` | reranker scores for all 948 x 100 pairs, job 56491 |
| `coverage_gtr_perq.jsonl`, `coverage_oracle_perq.jsonl` | per-question coverage |

The C1 rerun (`ctrl5`, job 55482) gave answers identical to `results/runs/asqa_cands_0_end.json` on 200 of 200
questions and is not stored again. The reranked input files (22 MB each) are rebuilt from the scores with
`build_reranked.py`; the checksum of the one used is in `retrieval/records/SELECTION.txt`.

The runs were made from a working folder outside the repo (`~/div1_diag`), so paths inside the stored outputs'
`args` and inside `retrieval/records/` show that folder. The scripts here are the same code with repo paths.

Not yet scored: ROUGE-L, MAUVE and citation recall/precision of the reranked runs (they need the T5-11B model).
