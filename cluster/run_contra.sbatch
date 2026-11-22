#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=contra
#SBATCH --output=logs/contra_%j.log
#SBATCH --error=logs/contra_%j.err
# Contradiction-rate diagnostic on an ASQA output file (no generation; minutes on one GPU).
# Runs twice: plain, and with the question as context. Outputs go to out/contra_<TAG>*.
#   sbatch --export=ALL,F=out/asqa_baseline.json,TAG=baseline cluster/run_contra.sbatch
cd "$SLURM_SUBMIT_DIR"
source cluster/env_alce.sh
: "${F:?set F=path/to/asqa_output.json}"
TAG=${TAG:-run}
python cluster/contradiction_rate.py "$F" --out_prefix "out/contra_${TAG}"
python cluster/contradiction_rate.py "$F" --out_prefix "out/contra_${TAG}_q" --with_question
