#!/bin/bash
# Entropy sweep (R1-R3) — re-runs the original --priv-loss entropy protocol from
# docs/superpowers/results/2026-04-18-itit-resnet18-no-noise-results.md
# under the corrected privacy package (firearm-filtered Stage C metric;
# Stage B in-loop privacy loss masks firearm samples).
#
# Per lambda: Stage B (30ep) -> convert ckpt -> refresh (2ep) -> Stage C
# (30ep) on both pre- and post-refresh checkpoints.
# Wall: ~3.5h per lambda x 3 lambdas = ~11h on 1x A100. 12h budget.

#SBATCH --job-name=NGT_Entropy_Sweep
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu

# --- USER-SPECIFIC ---
#SBATCH --output=Trained_Models/Entropy_Sweep/slurm_%j.out
# Uncomment + set if your cluster requires explicit account billing
# (e.g. you have multiple SU allocations, or sbatch rejects with "no account specified"):
##SBATCH --account=<YOUR_ACCOUNT>
# Uncomment + set if you want email notifications on job state changes:
##SBATCH --mail-type=ALL
##SBATCH --mail-user=<YOUR_EMAIL>
# --- end USER-SPECIFIC ---

set -euo pipefail

REPO_DIR="${REPO_DIR:-Over-The-Air-Neural-Group-Testing}"
ENV_NAME="${ENV_NAME:-ngt_env}"
MODULE_CONDA="${MODULE_CONDA:-Anaconda3/2024.02-1}"
MODULE_CUDA="${MODULE_CUDA:-CUDA/12.9.0}"
STORAGE_ROOT="${STORAGE_ROOT:-$SCRATCH}"       # $SCRATCH on TAMU; $WORK / $HOME on others

cd "$STORAGE_ROOT/$REPO_DIR"

ml purge
ml WebProxy 2>/dev/null || true   # may not exist on non-TAMU clusters; ignore failure
ml "$MODULE_CUDA"
ml "$MODULE_CONDA"

source "$(conda info --base)/etc/profile.d/conda.sh"
ENV_PREFIX="${ENV_PREFIX:-${SCRATCH:-/scratch/user/$USER}/conda_envs/${ENV_NAME}}"
if [[ -d "$ENV_PREFIX" ]]; then
    conda activate "$ENV_PREFIX"
else
    conda activate "$ENV_NAME"
fi

# `conda activate` ignores `set -e` when the env is missing — falls back to
# base python silently. Force-fail if torch isn't actually importable.
python -c "import torch, torchvision" || {
    echo "FATAL: torch/torchvision not importable in env '$ENV_NAME' (prefix: $ENV_PREFIX)." >&2
    echo "       Check: conda env list   (and set ENV_NAME=<your_env> or ENV_PREFIX=<full_path>)" >&2
    exit 1
}

mkdir -p Trained_Models/Entropy_Sweep

# Mount ImageNet squashfs so GroupTestingDataset symlinks resolve.
SQSH="${SCRATCH}/imagenet.sqsh"
IMGNET_MNT="${SCRATCH}/imagenet_mnt"
mkdir -p "$IMGNET_MNT"
squashfuse "$SQSH" "$IMGNET_MNT"
trap "fusermount -u '$IMGNET_MNT' 2>/dev/null || true" EXIT

# Sweep parameters (override via env if you want fewer epochs for a
# smoke test, e.g. STAGE_B_EPOCHS=1 STAGE_C_EPOCHS=1 LAMS=0.1 sbatch ...)
DATA="${DATA:-data/GroupTestingDataset}"
ARCH="${ARCH:-resnet18}"
STAGE_A_CKPT="${STAGE_A_CKPT:-Trained_Models/StageA_ITIT_ResNet18/model_best.pth.tar}"
STAGE_B_EPOCHS="${STAGE_B_EPOCHS:-30}"
STAGE_C_EPOCHS="${STAGE_C_EPOCHS:-30}"
REFRESH_EPOCHS="${REFRESH_EPOCHS:-2}"
REFRESH_LR="${REFRESH_LR:-0.0001}"
LAMS="${LAMS:-0.1 0.3 1.0}"
PRIV_LOSS="entropy"

if [[ ! -f "$STAGE_A_CKPT" ]]; then
    echo "FATAL: Stage A checkpoint not found at $STAGE_A_CKPT" >&2
    echo "Submit hprc_scripts/stage_a_resnet18.slurm first, or override STAGE_A_CKPT." >&2
    exit 1
fi

nvidia-smi

LOG="Trained_Models/Entropy_Sweep/sweep.log"
exec > >(tee -a "$LOG") 2>&1

for LAM in $LAMS; do
    TAG="lam${LAM}_${PRIV_LOSS}"
    OUT_B="Trained_Models/StageB_ITIT_ResNet18_${TAG}"
    OUT_REFRESH="Trained_Models/RefreshedB_${TAG}"
    OUT_C_PRE="Trained_Models/StageC_OnStageB_ITIT_ResNet18_${TAG}"
    OUT_C_POST="Trained_Models/StageC_OnRefreshedB_${TAG}"

    echo "=========================================="
    echo " lambda=$LAM | priv-loss=$PRIV_LOSS | $(date)"
    echo "=========================================="

    # 1. Stage B (recovery-epochs=0; we do refresh via main.py separately
    #    to match the entropy protocol in the results doc).
    echo "[1/4] Stage B"
    python -u -m privacy.train_privacy \
        --stage-a-ckpt "$STAGE_A_CKPT" \
        --data "$DATA" --task-num 2 --background-K 0 --GT-alg 1 \
        -a "$ARCH" \
        --priv-loss "$PRIV_LOSS" --lambda "$LAM" --k-adv 5 \
        --stage-b-epochs "$STAGE_B_EPOCHS" --recovery-epochs 0 \
        --batch-size 32 -j 8 -valj 4 \
        --output_dir "$OUT_B"

    STAGE_B_CKPT="$OUT_B/stage_b_final.pth.tar"

    # 2. Convert Stage B ckpt to main.py format for refresh.
    echo "[2/4] Convert ckpt"
    MAIN_FMT="$OUT_B/stage_b_final_main_format.pth.tar"
    python hprc_scripts/convert_stage_b_to_main.py \
        --in "$STAGE_B_CKPT" --out "$MAIN_FMT"

    # 3. Refresh: 2 epochs at low LR on firearm CE only (E + R both unfrozen).
    echo "[3/4] Refresh"
    python -u main.py \
        --resume "$MAIN_FMT" \
        --data "$DATA" \
        --task-num 2 --background-K 0 --GT-alg 1 \
        -a "$ARCH" \
        --epochs "$REFRESH_EPOCHS" --batch-size 32 --lr "$REFRESH_LR" \
        --output_dir "$OUT_REFRESH" \
        --log-name refresh.log

    REFRESHED_CKPT="$OUT_REFRESH/checkpoint.pth.tar"
    if [[ ! -f "$REFRESHED_CKPT" ]]; then
        REFRESHED_CKPT="$OUT_REFRESH/model_best.pth.tar"
    fi

    # 4. Stage C on BOTH pre-refresh and post-refresh checkpoints.
    echo "[4a/4] Stage C on Stage B (pre-refresh)"
    python -u -m privacy.eval_privacy \
        --stage-b-ckpt "$STAGE_B_CKPT" \
        --data "$DATA" --task-num 2 --background-K 0 --GT-alg 1 \
        -a "$ARCH" \
        --stage-c-epochs "$STAGE_C_EPOCHS" --batch-size 32 \
        --output_dir "$OUT_C_PRE"

    echo "[4b/4] Stage C on Refreshed (post-refresh)"
    python -u -m privacy.eval_privacy \
        --stage-b-ckpt "$REFRESHED_CKPT" \
        --data "$DATA" --task-num 2 --background-K 0 --GT-alg 1 \
        -a "$ARCH" \
        --stage-c-epochs "$STAGE_C_EPOCHS" --batch-size 32 \
        --output_dir "$OUT_C_POST"

    echo "DONE lambda=$LAM @ $(date):"
    echo "  Stage B leakage   : $OUT_C_PRE/leakage.json"
    echo "  Refreshed leakage : $OUT_C_POST/leakage.json"
done

echo "=========================================="
echo "All entropy-sweep runs complete @ $(date)."
echo "Leakage JSONs:"
echo "  Trained_Models/StageC_OnStageB_ITIT_ResNet18_lam*_entropy/leakage.json"
echo "  Trained_Models/StageC_OnRefreshedB_lam*_entropy/leakage.json"
echo "Each contains 'top1_background_acc' (threat-model leakage)"
echo "and 'top1_combined_acc' (legacy contaminated number) for auditability."
echo "=========================================="
