#!/usr/bin/env bash
# OTA-NGT env setup on TAMU HPRC Grace.
# Mirrors the ECEN743-SP26-A1-DP make_env.sh pattern (a known-working
# Grace recipe), with this project's deps swapped in.
#
# Run:  bash hprc_scripts/env_setup.sh

ml purge
ml WebProxy
ml CUDA/12.9.0
ml Anaconda3/2024.02-1

mkdir -p $SCRATCH/conda_pkgs
mkdir -p $SCRATCH/conda_envs

source $(conda info --base)/etc/profile.d/conda.sh

conda config --add pkgs_dirs $SCRATCH/conda_pkgs
conda config --add envs_dirs $SCRATCH/conda_envs

rm -rf ~/.local/lib/python3.11/site-packages/torch*

conda env remove -n ngt_env_modern -y
conda create -n ngt_env_modern python=3.11 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ngt_env_modern

mkdir -p $SCRATCH/conda_envs/ngt_env_modern/etc/conda/activate.d
cat << 'EOF' > $SCRATCH/conda_envs/ngt_env_modern/etc/conda/activate.d/env_vars.sh
export PYTHONNOUSERSITE=1
EOF

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install numpy pillow scikit-learn matplotlib pandas tqdm pytest
