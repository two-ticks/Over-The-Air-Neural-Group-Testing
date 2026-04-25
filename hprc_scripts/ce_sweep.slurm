#!/bin/bash
# CE sweep (R1-R3) — adds the --priv-loss ce column to the existing
# entropy results, mirroring the protocol from
# docs/superpowers/results/2026-04-18-itit-resnet18-no-noise-results.md.
#
# Per lambda: Stage B (30ep) -> convert ckpt -> refresh (2ep) -> Stage C
# (30ep) on both pre- and post-refresh checkpoints.
# Wall: ~3.5h per lambda x 3 lambdas = ~11h on 1x A100. 12h budget.

#SBATCH --job-name=NGT_CE_Sweep
#SBATCH --time=12:00:00
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
#SBATCH --output=Trained_Models/CE_Sweep/slurm_%j.out
# --- end USER-SPECIFIC ---

set -euo pipefail

REPO_DIR="${REPO_DIR:-NeuralGroupTesting}"
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

mkdir -p Trained_Models/CE_Sweep

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
PRIV_LOSS="ce"

if [[ ! -f "$STAGE_A_CKPT" ]]; then
    echo "FATAL: Stage A checkpoint not found at $STAGE_A_CKPT" >&2
    echo "Submit hprc_scripts/stage_a_resnet18.slurm first, or override STAGE_A_CKPT." >&2
    exit 1
fi

nvidia-smi

LOG="Trained_Models/CE_Sweep/sweep.log"
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
echo "All CE-sweep runs complete @ $(date)."
echo "Compare to entropy baseline at:"
echo "  docs/superpowers/results/2026-04-18-itit-resnet18-no-noise-results.md"
echo "  Trained_Models/StageC_OnStageB_ITIT_ResNet18_lam*_entropy/leakage.json"
echo "  Trained_Models/StageC_OnRefreshedB_lam*/leakage.json"
echo "=========================================="
