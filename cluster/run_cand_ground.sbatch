#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --exclude=TC2N01,TC2N02
#SBATCH --job-name=cand_ground
#SBATCH --output=logs/cand_ground_%j.log
#SBATCH --error=logs/cand_ground_%j.err
# D2 NLI grounding, GPU step: DeBERTa NLI of each candidate's own passage -> each sentence of the candidate.
# All three sets in one job (Set C output is saved but not analysed until the pre-registered test).
#   sbatch cluster/run_cand_ground.sbatch
cd "$SLURM_SUBMIT_DIR"
source cluster/env_alce.sh
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) on $(hostname)"
for SET in A B; do
  python grounding/cand_ground.py --f results/runs/asqa_cands_0_end.json \
    --split "critique/splits/asqa_${SET}.json" --out "results/runs/cand_ground_${SET}.jsonl" || exit 1
done
python grounding/cand_ground.py --f results/runs/asqa_cands_C.json \
  --split critique/splits/asqa_C.json --out results/runs/cand_ground_C.jsonl
