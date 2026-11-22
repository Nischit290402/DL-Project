# Combined system

The final system stacks every component that improved on C1 in its own held-out test: the retrieval division's
reranked passages, the critique division's full critique score and the grounding division's D2-v2 grounding term.
What is combined and why, the ablation ladder and the tests are fixed in `combined/PREREG.md` (written before the
combined GPU run).

| File | What it does | Where |
| --- | --- | --- |
| `combined/rerank_combined.py` | Re-picks answers with any subset of the components switched on; imports each division's code | CPU |
| `cluster/run_combined_prep.sbatch` | Scores the retrieval runs and every candidate on the reranked passages; grounding and relevance features | GPU, about 1 h |
| `combined/run_ladder.sh` | Builds ladder rows R0-R3, scores them by lookup, runs every pre-registered comparison | CPU, about 1 min |

## Steps

1. Cluster, login node, once: download the relevance model the D2-v2 gate uses.
   ```bash
   cd ~/DL-Project && source cluster/env.sh
   python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-reranker-base')"
   ```
2. Cluster: GPU preparation (writes `results/runs/combined/` and the retrieval runs' `.perq.jsonl`).
   ```bash
   sbatch cluster/run_combined_prep.sbatch
   ```
   The log must end with `COMBINED_PREP_OK`; `merge_exact_cit.py` must report 0 str-em/ROUGE mismatches.
3. Laptop: copy the outputs and run the ladder.
   ```bash
   scp -r '<you>@10.96.189.12:DL-Project/results/runs/combined' results/runs/
   scp '<you>@10.96.189.12:DL-Project/results/runs/retrieval/*.perq.jsonl' results/runs/retrieval/
   bash combined/run_ladder.sh
   ```
   `R1_check` must report "answer changed ... 0" for both sets.
4. Cluster: score the final answers directly (checks the lookup and gives MAUVE).
   ```bash
   scp out/combined/final_948.json '<you>@10.96.189.12:DL-Project/out/'      # from the laptop
   sbatch --export=ALL,F=out/final_948.json,FLAGS=--mauve cluster/run_critique_score.sbatch
   ```
   Its str-em and citation averages must equal those of `out/combined/R3_final_948.json.perq.jsonl`.

To dry-run step 3 without the GPU outputs, point the script at the GTR candidates (shows critique + grounding
without retrieval): set `OUT_DIR`, `PREP_DIR`, `CANDS_AB`, `CANDS_C` as described at the top of `run_ladder.sh`.
