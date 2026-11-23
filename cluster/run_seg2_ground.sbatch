#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=01:00:00
#SBATCH --exclude=TC2N01,TC2N02
#SBATCH --job-name=seg2_gr
#SBATCH --output=logs/seg2_ground_%j.log
#SBATCH --error=logs/seg2_ground_%j.err
# Support scores for second-segment continuations: whole-passage (grounding/cand_ground.py) and evidence-sentence
# (grounding/cand_ground_evidence.py), both unchanged.   sbatch --export=ALL,TAG=A,SPLIT=critique/splits/asqa_A.json cluster/run_seg2_ground.sbatch
cd "$SLURM_SUBMIT_DIR"
source cluster/env_alce.sh
set -eo pipefail
python grounding/cand_ground.py --f out/seg2_${TAG}.json --split "$SPLIT" --out out/seg2_${TAG}.ground.jsonl
python grounding/cand_ground_evidence.py --f out/seg2_${TAG}.json --split "$SPLIT" --out out/seg2_${TAG}.ground_ev.jsonl
