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
| `convert_stage_b_to_main.py` | Renames `state_dict_backbone → state_dict` so a Stage B checkpoint can be passed to `main.py --resume` for the refresh step. |

---

## Before you submit anything

### One-time

1. **Substitute placeholders.** Each `.slurm` file has these fields you
   must edit:
   - `#SBATCH --account=<YOUR_ACCOUNT>` → your cluster's billing/SU
     account (TAMU users: `myproject` on a login node)
   - `#SBATCH --mail-user=<YOUR_EMAIL>` → for `--mail-type=ALL` notifications
   - The `cd $SCRATCH/<REPO_DIR>` line — match where you cloned the repo
     (`$SCRATCH` is the typical TAMU/SLURM convention; on other sites it
     may be `$WORK`, `$STORAGE`, or just `~/`)

2. **Verify the module names.** The slurm files load
   `Anaconda3/2024.06`, `CUDA/12.4.0`, and (optionally) `WebProxy`. These
   are TAMU LMOD names; on other sites the modules may be named
   differently (e.g. `anaconda3`, `cuda/12.4`). Check with
   `module avail` on a login node and edit the `module load` lines.

3. **Pick a `--dist-url` port.** Each Stage A slurm submission needs a
   unique loopback port (`tcp://127.0.0.1:<port>`) — see `gotchas.md` →
   "dist-url port collisions". The default in `stage_a_resnet18.slurm` is
   `7110`; if you submit two Stage A jobs that may land on the same node,
   change one.

4. **Run `env_setup.sh` once.** Creates the `ngt_env_modern` conda env on
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
   module load CUDA/12.4.0 Anaconda3/2024.06   # adjust to your cluster
   source activate ngt_env_modern
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

# the actual experiment:
sbatch hprc_scripts/ce_sweep.slurm

# monitor:
squeue --me
tail -f Trained_Models/CE_Sweep/sweep.log
```

After the sweep finishes, the leakage numbers land in:

```
Trained_Models/StageC_OnStageB_ITIT_ResNet18_lam{0.1,0.3,1.0}_ce/leakage.json
Trained_Models/StageC_OnRefreshedB_lam{0.1,0.3,1.0}_ce/leakage.json
```

Compare against the entropy baseline at:

```
docs/superpowers/results/2026-04-18-itit-resnet18-no-noise-results.md §3-5
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

## Caveats

- `env_setup.sh` pins `torch==2.6.0+cu124`. If your cluster's CUDA
  modules move to a newer version, adjust the wheel index URL — the rest
  of the env spec should still apply.
- `ce_sweep.slurm` is a single serial job. To split across three jobs
  for faster wall clock, copy it three times and edit the `LAMS` line in
  each.
