#!/bin/bash
# Stage A — utility pretrain for ResNet-18, ITIT, no channel noise.
# Output: Trained_Models/StageA_ITIT_ResNet18/{checkpoint,model_best}.pth.tar
# Wall: ~30 min on 1x A100 (the L40 first-pass took ~20 min).
#
# Skip this job if you already have model_best.pth.tar from a prior run.

#SBATCH --job-name=NGT_StageA_R18
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu

# --- USER-SPECIFIC ---
#SBATCH --output=Trained_Models/StageA_ITIT_ResNet18/slurm_%j.out
# Uncomment + set if your cluster requires explicit account billing
# (e.g. you have multiple SU allocations, or sbatch rejects with "no account specified"):
##SBATCH --account=<YOUR_ACCOUNT>
# Uncomment + set if you want email notifications on job state changes:
##SBATCH --mail-type=ALL
##SBATCH --mail-user=<YOUR_EMAIL>
# --- end USER-SPECIFIC ---

set -euo pipefail

REPO_DIR="${REPO_DIR:-Over-The-Air-Neural-Group-Testing}"     # name of repo dir under your storage root
ENV_NAME="${ENV_NAME:-ngt_env_modern}"
MODULE_CONDA="${MODULE_CONDA:-Anaconda3/2024.02-1}"
MODULE_CUDA="${MODULE_CUDA:-CUDA/12.9.0}"
STORAGE_ROOT="${STORAGE_ROOT:-$SCRATCH}"       # $SCRATCH on TAMU; $WORK / $HOME on others

cd "$STORAGE_ROOT/$REPO_DIR"

ml purge
ml WebProxy 2>/dev/null || true   # may not exist on non-TAMU clusters; ignore failure
ml "$MODULE_CUDA"
ml "$MODULE_CONDA"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# `conda activate` ignores `set -e` when the env is missing — falls back to
# base python silently. Force-fail if torch isn't actually importable.
python -c "import torch, torchvision" || {
    echo "FATAL: torch/torchvision not importable in env '$ENV_NAME'." >&2
    echo "       conda activate may have silently fallen through to base." >&2
    echo "       Check: conda env list   (and pass ENV_NAME=<your_env> to sbatch)" >&2
    exit 1
}

mkdir -p Trained_Models/StageA_ITIT_ResNet18

nvidia-smi

CUDA_VISIBLE_DEVICES=0 python -u main.py \
    --data data/GroupTestingDataset \
    --task-num 2 --background-K 0 --GT-alg 1 \
    -a resnet18 --pretrained \
    --epochs 20 --batch-size 32 --lr 0.001 \
    --output_dir Trained_Models/StageA_ITIT_ResNet18 \
    --log-name stage_a.log

echo "Stage A done. model_best.pth.tar should be at Trained_Models/StageA_ITIT_ResNet18/model_best.pth.tar"
