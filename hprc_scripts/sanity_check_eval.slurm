#!/bin/bash
# Phase 1 sanity-check: re-run Stage C eval on existing Stage B + Refreshed
# checkpoints under the firearm-filtered metric (privacy/eval_privacy.py).
# Does NOT retrain Stage B — fastest path to see whether the firearm
# contamination was hiding a real background-only privacy signal.
#
# Per ckpt: Stage C 30ep ~1.5h on A100. With 3 lambdas x 2 (pre/post refresh)
# x 2 priv-loss forms (ce + entropy if both Stage B ckpts exist) = up to 12
# evals, ~18h. Default LAMS/PRIV_LOSSES set below; override via env.
#
# Outputs to Trained_Models/StageC_*_sanity/ so the original (contaminated)
# leakage.json files are not overwritten.

#SBATCH --job-name=NGT_Sanity_Eval
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu

# --- USER-SPECIFIC ---
#SBATCH --output=Trained_Models/CE_Sweep/sanity_slurm_%j.out
##SBATCH --account=<YOUR_ACCOUNT>
##SBATCH --mail-type=ALL
##SBATCH --mail-user=<YOUR_EMAIL>
# --- end USER-SPECIFIC ---

set -euo pipefail

REPO_DIR="${REPO_DIR:-Over-The-Air-Neural-Group-Testing}"
ENV_NAME="${ENV_NAME:-ngt_env}"
MODULE_CONDA="${MODULE_CONDA:-Anaconda3/2024.02-1}"
MODULE_CUDA="${MODULE_CUDA:-CUDA/12.9.0}"
STORAGE_ROOT="${STORAGE_ROOT:-$SCRATCH}"

cd "$STORAGE_ROOT/$REPO_DIR"

ml purge
ml WebProxy 2>/dev/null || true
ml "$MODULE_CUDA"
ml "$MODULE_CONDA"

source "$(conda info --base)/etc/profile.d/conda.sh"
ENV_PREFIX="${ENV_PREFIX:-${SCRATCH:-/scratch/user/$USER}/conda_envs/${ENV_NAME}}"
if [[ -d "$ENV_PREFIX" ]]; then
    conda activate "$ENV_PREFIX"
else
    conda activate "$ENV_NAME"
fi

python -c "import torch, torchvision" || {
    echo "FATAL: torch/torchvision not importable in env '$ENV_NAME' (prefix: $ENV_PREFIX)." >&2
    exit 1
}

mkdir -p Trained_Models/CE_Sweep

SQSH="${SCRATCH}/imagenet.sqsh"
IMGNET_MNT="${SCRATCH}/imagenet_mnt"
mkdir -p "$IMGNET_MNT"
squashfuse "$SQSH" "$IMGNET_MNT"
trap "fusermount -u '$IMGNET_MNT' 2>/dev/null || true" EXIT

DATA="${DATA:-data/GroupTestingDataset}"
ARCH="${ARCH:-resnet18}"
STAGE_C_EPOCHS="${STAGE_C_EPOCHS:-30}"
LAMS="${LAMS:-0.1 0.3 1.0}"
PRIV_LOSSES="${PRIV_LOSSES:-ce entropy}"
STAGE_A_CKPT="${STAGE_A_CKPT:-Trained_Models/StageA_ITIT_ResNet18/model_best.pth.tar}"

nvidia-smi

LOG="Trained_Models/CE_Sweep/sanity_check.log"
exec > >(tee -a "$LOG") 2>&1

eval_one () {
    local stage_b_ckpt="$1"
    local out_dir="$2"
    local label="$3"
    if [[ ! -f "$stage_b_ckpt" ]]; then
        echo "  SKIP $label — no checkpoint at $stage_b_ckpt"
        return 0
    fi
    if [[ -f "$out_dir/leakage.json" ]]; then
        echo "  SKIP $label — already evaluated at $out_dir/leakage.json"
        return 0
    fi
    echo "  RUN  $label (ckpt=$stage_b_ckpt, out=$out_dir)"
    python -u -m privacy.eval_privacy \
        --stage-b-ckpt "$stage_b_ckpt" \
        --data "$DATA" --task-num 2 --background-K 0 --GT-alg 1 \
        -a "$ARCH" \
        --stage-c-epochs "$STAGE_C_EPOCHS" --batch-size 32 \
        --output_dir "$out_dir"
}

# Stage A baseline: leakage on the no-privacy-trained encoder. Without this,
# we can't tell whether Stage B's λ-sweep actually reduces leakage from a
# baseline X or just produces low absolute numbers (which a 976-way fresh
# adversary on 30ep training might give regardless of encoder).
echo "=========================================="
echo " Stage A baseline (no privacy training) | $(date)"
echo "=========================================="
eval_one "$STAGE_A_CKPT" \
         "Trained_Models/StageC_OnStageA_ITIT_ResNet18_sanity" \
         "Stage A baseline (no privacy)"

for PL in $PRIV_LOSSES; do
    for LAM in $LAMS; do
        TAG="lam${LAM}_${PL}"
        STAGE_B_CKPT="Trained_Models/StageB_ITIT_ResNet18_${TAG}/stage_b_final.pth.tar"
        REFRESHED_CKPT="Trained_Models/RefreshedB_${TAG}/checkpoint.pth.tar"
        if [[ ! -f "$REFRESHED_CKPT" ]]; then
            REFRESHED_CKPT="Trained_Models/RefreshedB_${TAG}/model_best.pth.tar"
        fi

        OUT_C_PRE_SANITY="Trained_Models/StageC_OnStageB_ITIT_ResNet18_${TAG}_sanity"
        OUT_C_POST_SANITY="Trained_Models/StageC_OnRefreshedB_${TAG}_sanity"

        echo "=========================================="
        echo " priv-loss=$PL lambda=$LAM | $(date)"
        echo "=========================================="

        eval_one "$STAGE_B_CKPT" "$OUT_C_PRE_SANITY" "Stage B (pre-refresh)  $TAG"
        eval_one "$REFRESHED_CKPT" "$OUT_C_POST_SANITY" "Refreshed (post)       $TAG"
    done
done

echo "=========================================="
echo "Sanity-check evals complete @ $(date)."
echo "Pull leakage JSONs locally with the rsync command in hprc_scripts/README.md."
echo "Inspect 'top1_background_acc' (the threat-model leakage) vs 'top1_combined_acc'"
echo "(the legacy contaminated number) in each leakage.json."
echo "=========================================="
