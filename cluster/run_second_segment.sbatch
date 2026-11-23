#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=03:00:00
#SBATCH --exclude=TC2N01,TC2N02
#SBATCH --job-name=seg2
#SBATCH --output=logs/seg2_%j.log
#SBATCH --error=logs/seg2_%j.err
# Second-segment continuations for the combined system's answers (self-rag/retrieval_lm/run_second_segment.py).
#   sbatch --export=ALL,TAG=AB,IDX=critique/splits/asqa_A.json,critique/splits/asqa_B.json,EXTRA= cluster/run_second_segment.sbatch
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh
mkdir -p out logs
ROOT=$(pwd)
IDX=$(echo "$IDX" | sed "s#\([^,]*\)#$ROOT/\1#g")
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) on $(hostname)"
cd self-rag/retrieval_lm
python run_second_segment.py --model_name "$SELFRAG_MODEL" --download_dir "$HF_HOME" \
  --picks "$ROOT/${PICKS:-results/runs/combined/final_948.json}" \
  --cands "$ROOT/results/runs/retrieval/asqa_rrR2K20_0_end.json,$ROOT/results/runs/retrieval/asqa_rrR2K20C_0_end.json" \
  --indices_file "$IDX" --output_file "$ROOT/out/seg2_${TAG}.json" ${EXTRA:-}
