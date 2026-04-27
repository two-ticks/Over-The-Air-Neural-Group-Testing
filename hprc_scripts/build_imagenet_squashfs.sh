#!/usr/bin/env bash
# Build a SquashFS image of ImageNet from the cluster's shared tar bundle.
#
# Run on an INTERACTIVE COMPUTE NODE (not a login node — extraction is
# heavy IO and the login watchdog will kill it):
#
#     srun --time=02:00:00 --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 \
#          --mem=32G --partition=short --pty bash
#
# Then:
#     bash hprc_scripts/build_imagenet_squashfs.sh
#
# Output:
#     $SCRATCH/imagenet.sqsh    (~140 GB compressed; one inode of yours)
#
# Reusable across all subsequent training jobs via squashfuse mount.

set -euo pipefail

# --- defaults; override via env ---
TAR_SRC="${TAR_SRC:-/scratch/data/imagenet_prepared.tar}"
SQSHFS_DST="${SQSHFS_DST:-$SCRATCH/imagenet.sqsh}"
WORK_DIR="${WORK_DIR:-${TMPDIR:-/tmp}/imagenet_build_$$}"
COMPRESS="${COMPRESS:-lz4}"
COMPRESS_LEVEL="${COMPRESS_LEVEL:-9}"

# --- pre-flight ---
echo "[1/5] Pre-flight checks"

if [[ ! -f "$TAR_SRC" ]]; then
    echo "FATAL: source tar not found at $TAR_SRC" >&2
    exit 1
fi

if ! command -v mksquashfs >/dev/null 2>&1; then
    echo "FATAL: mksquashfs not found. Try: module load squashfs-tools" >&2
    echo "(or load whichever module on this cluster provides it)" >&2
    exit 1
fi

# Confirm we have enough free space in the work dir for extraction (~150 GB)
WORK_PARENT="$(dirname "$WORK_DIR")"
AVAIL_KB=$(df -P "$WORK_PARENT" | awk 'NR==2 {print $4}')
AVAIL_GB=$((AVAIL_KB / 1024 / 1024))
if (( AVAIL_GB < 160 )); then
    echo "WARNING: only ${AVAIL_GB} GB available at $WORK_PARENT — extraction needs ~150 GB." >&2
    echo "Override WORK_DIR= to point at a bigger filesystem if this fails." >&2
fi

mkdir -p "$WORK_DIR"

# --- extract ---
echo "[2/5] Extracting $TAR_SRC -> $WORK_DIR (this is the slow step, ~5-15 min)"
tar -C "$WORK_DIR" -xf "$TAR_SRC"

# Sanity-check the extracted layout
if [[ ! -d "$WORK_DIR/imagenet/train" ]] || [[ ! -d "$WORK_DIR/imagenet/val" ]]; then
    echo "FATAL: expected $WORK_DIR/imagenet/{train,val} after extraction" >&2
    echo "Got:" >&2
    ls -la "$WORK_DIR" >&2
    exit 1
fi

NUM_TRAIN_WNIDS=$(ls "$WORK_DIR/imagenet/train" | wc -l)
NUM_VAL_WNIDS=$(ls "$WORK_DIR/imagenet/val" | wc -l)
echo "    train wnids: $NUM_TRAIN_WNIDS"
echo "    val   wnids: $NUM_VAL_WNIDS"
if (( NUM_TRAIN_WNIDS < 990 )); then
    echo "WARNING: expected ~1000 train wnids, got $NUM_TRAIN_WNIDS — proceeding anyway" >&2
fi

# --- build squashfs ---
echo "[3/5] Building squashfs ($COMPRESS, level $COMPRESS_LEVEL) -> $SQSHFS_DST"
echo "    this takes 20-40 minutes; lots of small files to compress"

mkdir -p "$(dirname "$SQSHFS_DST")"
# -no-progress for cleaner output in slurm logs; remove if you want a progress bar
mksquashfs "$WORK_DIR/imagenet" "$SQSHFS_DST" \
    -comp "$COMPRESS" -Xcompression-level "$COMPRESS_LEVEL" \
    -no-progress -noappend

# --- verify ---
echo "[4/5] Verifying squashfs"
SQSHFS_SIZE=$(ls -lh "$SQSHFS_DST" | awk '{print $5}')
echo "    $SQSHFS_DST  ($SQSHFS_SIZE)"

if command -v squashfuse >/dev/null 2>&1; then
    MNT="${WORK_DIR}_verify"
    mkdir -p "$MNT"
    squashfuse "$SQSHFS_DST" "$MNT"
    VERIFY_TRAIN=$(ls "$MNT/train" | wc -l)
    VERIFY_VAL=$(ls "$MNT/val" | wc -l)
    fusermount -u "$MNT"
    rmdir "$MNT"
    echo "    mount verified: train=$VERIFY_TRAIN wnids, val=$VERIFY_VAL wnids"
else
    echo "    squashfuse not available; skipping mount verification"
fi

# --- cleanup ---
echo "[5/5] Cleanup"
rm -rf "$WORK_DIR"
echo "    removed $WORK_DIR"

cat <<EOF

Build complete.
SquashFS:  $SQSHFS_DST  ($SQSHFS_SIZE)

To use in slurm jobs, add this near the top of the script (after module load):

    MOUNT="\$TMPDIR/imagenet_mnt"
    mkdir -p "\$MOUNT"
    squashfuse "$SQSHFS_DST" "\$MOUNT"
    trap "fusermount -u \$MOUNT" EXIT
    mkdir -p data
    ln -sfn "\$MOUNT" data/ImageNet-ILSVRC2012

Then build data/GroupTestingDataset/ once via data_scripts/create_dataset_from_imagenet.py
(same as before — symlinks read through the squashfs transparently).

EOF
