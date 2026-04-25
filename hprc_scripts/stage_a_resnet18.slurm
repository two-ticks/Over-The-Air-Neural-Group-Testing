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

# --- USER-SPECIFIC: edit these before submitting ---
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<YOUR_EMAIL>
#SBATCH --output=Trained_Models/StageA_ITIT_ResNet18/slurm_%j.out
# --- end USER-SPECIFIC ---

set -euo pipefail

REPO_DIR="${REPO_DIR:-NeuralGroupTesting}"     # name of repo dir under your storage root
ENV_NAME="${ENV_NAME:-ngt_env_modern}"
MODULE_CONDA="${MODULE_CONDA:-Anaconda3/2024.06}"
MODULE_CUDA="${MODULE_CUDA:-CUDA/12.4.0}"
STORAGE_ROOT="${STORAGE_ROOT:-$SCRATCH}"       # $SCRATCH on TAMU; $WORK / $HOME on others

cd "$STORAGE_ROOT/$REPO_DIR"

module purge
module load "$MODULE_CONDA"
module load "$MODULE_CUDA"

# Optional outbound HTTP proxy. Set HTTP_PROXY/HTTPS_PROXY in your
# environment before sbatch if your cluster requires it (and add
# `module load WebProxy` or equivalent above). Leave unset on sites with
# direct internet.
if [[ -n "${HTTP_PROXY:-}" ]]; then
    export http_proxy="$HTTP_PROXY"
    export https_proxy="${HTTPS_PROXY:-$HTTP_PROXY}"
fi

source activate "$ENV_NAME"

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
