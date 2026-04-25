#!/usr/bin/env bash
# One-time env setup for OTA-NGT + privacy package on an HPRC cluster.
#
# Run on a LOGIN NODE or interactive compute node. Idempotent — safe to
# re-run; old env is removed first.
#
# This is built on the working pattern from another rl_env project on
# the same cluster: explicit pkgs/envs dirs in $SCRATCH (so $HOME's
# small quota isn't a problem), WebProxy module for outbound HTTP,
# PYTHONNOUSERSITE=1 activation hook to stop ~/.local/ shadowing.
#
# Cluster-specific overrides via env vars:
#   MODULE_CONDA, MODULE_CUDA — LMOD module names (defaults match Grace)
#   ENV_NAME                  — conda env name
#   PY_VER                    — python version
#   TORCH_INDEX               — pip index URL for torch wheels (must
#                                match MODULE_CUDA's runtime)

set -euo pipefail

ENV_NAME="${ENV_NAME:-ngt_env_modern}"
PY_VER="${PY_VER:-3.10}"
MODULE_CONDA="${MODULE_CONDA:-Anaconda3/2024.02-1}"
MODULE_CUDA="${MODULE_CUDA:-CUDA/12.4.0}"
# Default cu124 to match the torch 2.6.0+cu124 from the existing L40
# first-pass results. If you switch MODULE_CUDA to 12.8.0/12.9.0,
# also switch TORCH_INDEX to the cu128 URL below.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.21.0}"

# 1. Modules. WebProxy auto-exports http_proxy/https_proxy on TAMU HPRC.
echo "[1/7] Loading modules"
ml purge
ml WebProxy
ml "$MODULE_CUDA"
ml "$MODULE_CONDA"

# 2. Point conda at $SCRATCH for both packages and envs (NOT $HOME — quota too small).
echo "[2/7] Configuring conda dirs in \$SCRATCH"
mkdir -p "$SCRATCH/conda_pkgs"
mkdir -p "$SCRATCH/conda_envs"

source "$(conda info --base)/etc/profile.d/conda.sh"

# add_paths is idempotent — if the entry already exists, conda silently no-ops.
conda config --add pkgs_dirs "$SCRATCH/conda_pkgs" 2>/dev/null || true
conda config --add envs_dirs "$SCRATCH/conda_envs" 2>/dev/null || true

# 3. Clean any stale torch in user-site that could shadow the env's torch.
echo "[3/7] Cleaning user-site torch shadows"
rm -rf ~/.local/lib/python*/site-packages/torch* 2>/dev/null || true

# 4. Create env (remove first if exists, for idempotency).
echo "[4/7] Creating conda env '$ENV_NAME' (python=$PY_VER)"
conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
conda create -n "$ENV_NAME" "python=$PY_VER" -y

# 5. Activate.
echo "[5/7] Activating env"
conda activate "$ENV_NAME"

# 6. Activation hook: stop ~/.local/ packages from leaking into env.
echo "[6/7] Writing PYTHONNOUSERSITE activation hook"
ACTIVATE_DIR="$SCRATCH/conda_envs/$ENV_NAME/etc/conda/activate.d"
mkdir -p "$ACTIVATE_DIR"
cat << 'EOF' > "$ACTIVATE_DIR/env_vars.sh"
export PYTHONNOUSERSITE=1
EOF
export PYTHONNOUSERSITE=1

# 7. Install torch + the small extras the privacy code needs.
echo "[7/7] Installing torch + extras"
pip install --upgrade pip
pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" --index-url "$TORCH_INDEX"
pip install \
    numpy \
    pillow \
    scikit-learn \
    matplotlib \
    pandas \
    tqdm \
    pytest

# Sanity check.
python -c "import torch, torchvision; print('torch', torch.__version__, 'tv', torchvision.__version__, 'cuda_avail', torch.cuda.is_available())"

cat <<EOF

Done. Reactivate later in any shell with:
  ml $MODULE_CONDA $MODULE_CUDA
  source \$(conda info --base)/etc/profile.d/conda.sh
  conda activate $ENV_NAME

EOF
