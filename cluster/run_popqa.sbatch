#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=04:00:00
#SBATCH --job-name=selfrag_popqa
#SBATCH --output=logs/popqa_%j.log
#SBATCH --error=logs/popqa_%j.err
# PopQA, adaptive retrieval, settings from the upstream README (paper: 54.9 for 7B).
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh
mkdir -p out
nvidia-smi --query-gpu=name,memory.total --format=csv
cd self-rag/retrieval_lm
python run_short_form.py \
  --model_name "$SELFRAG_MODEL" \
  --download_dir "$HF_HOME" \
  --input_file ../../data/eval_data/popqa_longtail_w_gs.jsonl \
  --mode adaptive_retrieval \
  --max_new_tokens 100 \
  --threshold 0.2 \
  --output_file ../../out/popqa_adaptive.json \
  --metric match --ndocs 10 \
  --use_groundness --use_utility --use_seqscore \
  --dtype half
