# `hprc_scripts/` — HPRC slurm scripts

Self-contained SLURM tooling for running OTA-NGT and the privacy-package
sweeps on a typical HPRC cluster (login + compute partition with GPUs,
LMOD modules, conda, optional outbound HTTP proxy).

Tested with TAMU HPRC's Grace cluster as the reference environment, but
the cluster-specific values (account, email domain, module names, proxy
IP) are pulled out as placeholders or environment variables — adapt for
your own site.

---

## Files

| File | Purpose |
| ---- | ------- |
| `env_setup.sh` | One-time login-node conda env creation. Run once per cluster account. |
| `stage_a_resnet18.slurm` | Stage A utility pretrain (ResNet-18, ITIT, no noise). Skip if you already have `Trained_Models/StageA_ITIT_ResNet18/model_best.pth.tar`. |
| `ce_sweep.slurm` | The R1–R3 sweep — `--priv-loss ce` at λ ∈ {0.1, 0.3, 1.0}, end-to-end (Stage B + 2-ep refresh + Stage C pre-/post-refresh). |
| `entropy_sweep.slurm` | Same protocol as `ce_sweep.slurm` but with `--priv-loss entropy`. Run alongside the CE sweep for apples-to-apples comparison under the firearm-filtered Stage C metric. |
| `sanity_check_eval.slurm` | Re-runs Stage C eval against existing Stage B + refreshed checkpoints, using the firearm-filtered metric in `privacy/eval_privacy.py`. Cheaper than re-running Stage B; produces `Trained_Models/StageC_*_sanity/leakage.json` with `top1_background_acc` (threat-model leakage) alongside the legacy `top1_combined_acc`. |
| `convert_stage_b_to_main.py` | Renames `state_dict_backbone → state_dict` so a Stage B checkpoint can be passed to `main.py --resume` for the refresh step. |

---

## Before you submit anything

### One-time

1. **Substitute placeholders.** The slurm files come with most of the
   user-specific bits already commented out as opt-in fallbacks:
   - The `cd $SCRATCH/<REPO_DIR>` line — match where you cloned the repo
     (`$SCRATCH` is the typical TAMU/SLURM convention; on other sites it
     may be `$WORK`, `$STORAGE`, or just `~/`).
   - **`#SBATCH --account=...` is not set by default.** Most users with
     a single SU allocation will have slurm pick it automatically.
     Uncomment the `##SBATCH --account=<YOUR_ACCOUNT>` fallback line if
     `sbatch` rejects with "no account specified" or you have multiple
     allocations to choose from. Find your account with `myproject`
     (TAMU) or `sacctmgr show user $USER` (generic).
   - **Email notifications are off by default.** Slurm sends no mail
     unless `--mail-type` is set. If you do want notifications,
     uncomment both `##SBATCH --mail-type=ALL` and
     `##SBATCH --mail-user=<YOUR_EMAIL>` fallback lines.

2. **Verify the module names.** The slurm files load
   `Anaconda3/2024.02-1`, `CUDA/12.9.0`, and (optionally) `WebProxy`.
   These are TAMU Grace LMOD names as of 2026-04; on other sites or if
   Grace updates its modules they may be named differently. Check with
   `module avail Anaconda` and `module avail CUDA` on a login node and
   override via `MODULE_CONDA=...` / `MODULE_CUDA=...` at submission, or
   edit the defaults in the slurm files.

3. **Pick a `--dist-url` port.** Each Stage A slurm submission needs a
   unique loopback port (`tcp://127.0.0.1:<port>`) — see `gotchas.md` →
   "dist-url port collisions". The default in `stage_a_resnet18.slurm` is
   `7110`; if you submit two Stage A jobs that may land on the same node,
   change one.

4. **Run `env_setup.sh` once.** Creates the `ngt_env` conda env on
   your storage. The privacy package needs torch 2.6+ (the committed
   `requirements.{txt,yml}` files are for the older 1.8.1 codebase and
   are stale for the current code path).

5. **Outbound proxy (optional).** Some HPRC clusters firewall compute
   nodes off the public internet and require an HTTP proxy for `pip` /
   `torch.hub` fetches. If yours does, set `HTTP_PROXY` and
   `HTTPS_PROXY` env vars before `sbatch`-ing (the slurm files honor
   them). If your compute nodes have direct internet, leave both unset.

### Per-experiment

6. **Verify the dataset is built.** `$SCRATCH/<REPO_DIR>/data/GroupTestingDataset/{0,1}/{train,val}/<wnid>/`
   must exist. See `data_scripts/` for the prep scripts (raw ImageNet
   → GroupTestingDataset).

7. **Smoke test before queueing the long job.** Run for one batch on an
   interactive node to verify the env, GPU, and dataset are wired up:

   ```bash
   srun --time=00:30:00 --cpus-per-task=8 --mem=32G \
        --gres=gpu:a100:1 --partition=gpu --pty bash
   module load CUDA/12.9.0 Anaconda3/2024.02-1   # adjust to your cluster
   source activate ngt_env
   cd $SCRATCH/<REPO_DIR>
   pytest tests/ -v -m "not slow"
   ```

---

## Quickstart

```bash
# (one-time) on a login node:
bash hprc_scripts/env_setup.sh

# (only if you don't already have a Stage A checkpoint):
sbatch hprc_scripts/stage_a_resnet18.slurm

# the actual experiment — submit immediately after Stage A, run when it finishes:
STAGE_A_JOB=$(sbatch --parsable hprc_scripts/stage_a_resnet18.slurm)
sbatch --dependency=afterok:$STAGE_A_JOB hprc_scripts/ce_sweep.slurm

# or if you already submitted Stage A and have its job ID:
sbatch --dependency=afterok:<stage_a_jobid> hprc_scripts/ce_sweep.slurm

# monitor:
squeue --me
tail -f Trained_Models/CE_Sweep/sweep.log
```

`--parsable` makes `sbatch` print only the job ID (no "Submitted batch job …" text), so
it can be captured into a shell variable. `--dependency=afterok:<id>` holds the CE sweep
in the queue until Stage A exits cleanly; SLURM cancels it automatically if Stage A
fails. Use `afterany` instead of `afterok` to run the sweep regardless of Stage A's exit
status.

After the sweep finishes, the leakage numbers land in:

```
Trained_Models/StageC_OnStageB_ITIT_ResNet18_lam{0.1,0.3,1.0}_ce/leakage.json
Trained_Models/StageC_OnRefreshedB_lam{0.1,0.3,1.0}_ce/leakage.json
```

Compare against the entropy baseline at:

```
docs/superpowers/results/2026-04-18-itit-resnet18-no-noise-results.md §3-5
```

To pull just the `leakage.json` files to your laptop (run from a local
shell, not the login node):

```bash
rsync -avm --include='*/' --include='leakage.json' --exclude='*' <NETID>@grace.hprc.tamu.edu:/scratch/user/<NETID>/<REPO_DIR>/Trained_Models/ ./Trained_Models_leakage/
```

---

## Compute budget

ResNet-18 ITIT no-noise on 1× A100 (similar throughput to the L40 used
for the entropy first-pass):

| Phase                   | Per λ | × 3 λ |
| ----------------------- | ----- | ----- |
| Stage B (30 epochs)     | ~2 h  | 6 h   |
| Refresh (2 epochs)      | ~10 min | 30 min |
| Stage C × 2 (30 epochs each) | ~1.5 h | 4.5 h |
| **Total**               | ~3.5 h | **~11 h** |

`ce_sweep.slurm` requests `--time=12:00:00` to fit. If your queue is
backed up, split into three λ-specific jobs (one per λ) — finishes in
~3.5 h wall-clock if all three start at once.

---

## Building `imagenet.sqsh`

Run `build_imagenet_squashfs.sh` on an **interactive compute node** (not
a login node — extraction is heavy IO):

```bash
srun --time=02:00:00 --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 \
     --mem=32G --partition=gpu --pty bash
bash $SCRATCH/Over-The-Air-Neural-Group-Testing/hprc_scripts/build_imagenet_squashfs.sh
```

**Do not override `WORK_DIR` to a `$SCRATCH` path.** The script defaults
`WORK_DIR` to `$TMPDIR` (e.g. `/tmp/job.<jobid>`) which on Grace A100
nodes is a ~1.5 TB NVMe drive with no inode quota. `$SCRATCH` (Lustre)
has a 250k-file inode limit; ImageNet contains ~1.28 M files, so
extraction to Lustre always fails. The error from `tar` reads "Disk quota
exceeded" but the root cause is the inode count, not disk space.

**Compressor must be one Grace's `mksquashfs` supports.** The script
defaults to `lz4`. Grace's squashfs-tools do not include `zstd`; available
compressors are `gzip` (default upstream), `lzma`, `lzo`, `lz4`, `xz`.
`lz4` is preferred: JPEGs are already compressed so ratio differences are
negligible, and lz4 decompresses fastest during training IO. If the
`mksquashfs` step fails mid-run (after extraction completes), run it
directly on the already-extracted directory to avoid re-extracting:

```bash
mksquashfs /tmp/job.<JOBID>/imagenet_build_<PID>/imagenet \
    $SCRATCH/imagenet.sqsh -comp lz4 -no-progress -noappend
```

On Grace, `$TMPDIR` is set automatically by SLURM to
`/tmp/job.<SLURM_JOB_ID>`. Always confirm before running:

```bash
echo "TMPDIR=$TMPDIR"
df -h "$TMPDIR"   # should show /dev/mapper/nvme-* with ~1.5T free
```

---

## Caveats

- `env_setup.sh` pins torch to the cu128 wheel index. If your cluster's
  CUDA modules change, adjust the `--index-url` in the pip install line.
- `ce_sweep.slurm` is a single serial job. To split across three jobs
  for faster wall clock, copy it three times and edit the `LAMS` line in
  each.
