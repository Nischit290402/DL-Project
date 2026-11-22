# PopQA baseline (Self-RAG 7B, adaptive retrieval)

Run: 2026-11-05, NTU CCDS TC2, job 51213, node TC2N04 (NVIDIA L40S 46 GB), 38 min 54 s wall.
Command: `cluster/run_popqa.sbatch` (upstream README settings: `--mode adaptive_retrieval --threshold 0.2 --ndocs 10 --max_new_tokens 100 --use_groundness --use_utility --use_seqscore --dtype half`).
Data: `data/eval_data/popqa_longtail_w_gs.jsonl`, 1399 questions, pre-retrieved contexts.
Env: `cluster/requirements-lock.txt` (vllm 0.2.6, torch 2.1.2+cu121, transformers 4.36.2).

| Metric | This run | Paper (Table 2, Self-RAG 7B) |
| --- | --- | --- |
| PopQA match accuracy | 54.97 | 54.9 |

Raw predictions: `out/popqa_adaptive.json` on the cluster (`~/DL-Project/out/`, not committed).

Notes:
- The script prints `Retrieval Frequencies: 174.875`. That line divides the retrieval count by the number of keys in the results dict (8) instead of the number of questions; 174.875 x 8 = 1399, so retrieval was triggered for every question at threshold 0.2. Left as is in upstream code; worth a look when we touch the retrieval decision.
- Throughput: about 36 questions/min with 10 passages each.
