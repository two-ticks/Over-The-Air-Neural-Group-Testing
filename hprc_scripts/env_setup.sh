#!/usr/bin/env bash
# One-time env setup for OTA-NGT + privacy package on an HPRC cluster.
#
# Run on a LOGIN NODE (env creation is allowed there; training is not).
# Creates a conda env named ngt_env_modern with the modern torch stack
# the current code path was actually tested against (torch 2.6+cu124,
# torchvision 0.21, sklearn 1.8) — the committed requirements.{txt,yml}
# pin the older 1.8.1 codebase and don't apply.
#
# Idempotent: re-running just verifies the env exists and is importable.
#
# Cluster-specific bits to adjust before running:
#   * MODULE_CONDA / MODULE_CUDA — LMOD module names; defaults are
#     TAMU-style. Check `module avail` on a login node.
#   * HTTP_PROXY / HTTPS_PROXY — set these in your shell BEFORE running
#     this script if your cluster firewalls compute nodes / login nodes
#     off direct internet. Otherwise leave both unset.

set -euo pipefail

ENV_NAME="${ENV_NAME:-ngt_env_modern}"
PY_VER="${PY_VER:-3.10}"
MODULE_CONDA="${MODULE_CONDA:-Anaconda3/2024.06}"
MODULE_CUDA="${MODULE_CUDA:-CUDA/12.4.0}"

# 1. Load modules. Adjust the MODULE_* defaults if your cluster names them
#    differently (e.g. `anaconda3`, `cuda/12.4`).
echo "[1/4] Loading modules"
module purge
module load "$MODULE_CONDA"     # provides conda
module load "$MODULE_CUDA"      # CUDA toolkit; must match torch wheel below

# Optional outbound HTTP proxy. Some HPRC sites firewall login/compute
# nodes off the public internet and require pip / torch.hub fetches to go
# through a proxy. If your site does, export HTTP_PROXY/HTTPS_PROXY in
# your shell before running this script (and possibly load a `WebProxy`
# module). On sites with direct internet, leave them unset.
if [[ -n "${HTTP_PROXY:-}" ]]; then
    export http_proxy="$HTTP_PROXY"
    export https_proxy="${HTTPS_PROXY:-$HTTP_PROXY}"
    echo "    using proxy: $http_proxy"
fi

# 2. Create the env if missing.
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[2/4] Env '$ENV_NAME' already exists, skipping creation"
else
    echo "[2/4] Creating conda env '$ENV_NAME' (python=$PY_VER)"
    conda create -y -n "$ENV_NAME" "python=$PY_VER"
fi

# 3. Install the modern torch stack into the env.
echo "[3/4] Installing torch + deps into '$ENV_NAME'"
source activate "$ENV_NAME"

pip install --upgrade pip

# Pin to the versions the first-pass results were produced with. Update
# only with eyes open — a torch major bump can move the encode/channel/
# decode interface in subtle ways.
pip install \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

pip install \
    numpy \
    pillow \
    scikit-learn \
    matplotlib \
    pandas \
    tqdm \
    pytest

# 4. Sanity check: import torch, confirm CUDA available (only meaningful
#    if you happen to be on a node with a GPU; on a login node without
#    GPUs this prints False — that's fine, the actual GPU check happens
#    inside the slurm jobs).
echo "[4/4] Sanity import"
python -c "import torch, torchvision; print('torch', torch.__version__, 'tv', torchvision.__version__, 'cuda_avail', torch.cuda.is_available())"

echo
echo "Done. Activate with:"
echo "  module load $MODULE_CUDA $MODULE_CONDA"
echo "  source activate $ENV_NAME"
