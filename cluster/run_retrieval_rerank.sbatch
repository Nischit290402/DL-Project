#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=01:00:00
#SBATCH --exclude=TC2N01,TC2N02
#SBATCH --job-name=d1_rerank
#SBATCH --output=logs/rerank_%j.log
# Retrieval division: score every (question, passage) pair of the ASQA top-100 file with two cross-encoder
# rerankers, then build the four input files allowed by retrieval/PREREG.md. Submit from the repo root.
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh
IN=data/eval_data/asqa_eval_gtr_top100.json
OUT=results/runs/retrieval; mkdir -p $OUT data/reranked
python retrieval/rerank_score.py $IN cross-encoder/ms-marco-MiniLM-L-12-v2 $OUT/scores_R1.jsonl
python retrieval/rerank_score.py $IN BAAI/bge-reranker-v2-m3 $OUT/scores_R2.jsonl
for R in R1 R2; do for K in 20 100; do
  python retrieval/build_reranked.py $IN $OUT/scores_$R.jsonl $K data/reranked/asqa_${R}_K$K.json
done; done
sha256sum $OUT/scores_R*.jsonl data/reranked/asqa_R*_K*.json
