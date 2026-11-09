# source this before scoring: source cluster/env_alce.sh
# Same conda env as inference (scoring packages are added to it by setup_alce.sh).
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export HF_HUB_OFFLINE=1          # compute nodes: use the cache filled by setup_alce.sh
export ALCE_DIR=${ALCE_DIR:-$HOME/ALCE}
