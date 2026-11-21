#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --exclude=TC2N01,TC2N02
#SBATCH --job-name=nli_probe
#SBATCH --output=logs/nli_probe_%j.log
#SBATCH --error=logs/nli_probe_%j.err
# Re-score DeBERTa's near-verbatim failures under controlled variations (grounding/nli_probe.py).
#   sbatch cluster/run_nli_probe.sbatch
cd "$SLURM_SUBMIT_DIR"
source cluster/env_alce.sh
python -c "import transformers, tokenizers, sentencepiece, torch; print(transformers.__version__, tokenizers.__version__, sentencepiece.__version__, torch.__version__)"
python grounding/nli_probe.py --fail results/runs/verbatim_fail_AB.jsonl --control results/runs/verbatim_pass_AB.jsonl \
  --out results/runs/nli_probe.json
