# source this before scoring: source cluster/env_alce.sh
module load miniconda3/25.5.1 >/dev/null 2>&1 || true
eval "$(conda shell.bash hook)"
conda activate $HOME/envs/alce
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export HF_HUB_OFFLINE=1          # compute nodes: use the cache filled by setup_alce.sh
export TOKENIZERS_PARALLELISM=false
export ALCE_DIR=${ALCE_DIR:-$HOME/ALCE}
