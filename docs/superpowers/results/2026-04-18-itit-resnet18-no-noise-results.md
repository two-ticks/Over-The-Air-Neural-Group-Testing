# First-pass results: ITIT, ResNet-18, no-noise

**Date:** 2026-04-18
**Status:** First exploratory pass complete. Results expose a fragility in the privacy mechanism at this arch; bigger backbones / GTGT-FM not yet tested.
**Spec:** `docs/superpowers/specs/2026-04-17-privacy-preserving-ota-ngt-design.md`
**Plan:** `docs/superpowers/plans/2026-04-17-privacy-preserving-ota-ngt.md`

> **Superseded notice (2026-04-28):** The Stage C leakage numbers in this
> document were computed before the firearm-filter fix in
> `privacy/eval_privacy.py` and `privacy/trainer.py`. Both the in-loop
> Stage B privacy loss and the Stage C eval included firearm-class
> samples in the adversary's data, conflating the receiver's legitimate
> firearm-vs-background output with actual privacy leakage. The 35%
> baseline plateau is dominated by firearm-wnid classification accuracy,
> not by background-class leakage. Re-run both `--priv-loss entropy`
> and `--priv-loss ce` sweeps via `hprc_scripts/{entropy,ce}_sweep.slurm`
> on the corrected pipeline; the new leakage JSONs include
> `top1_background_acc` (threat-model leakage) and `top1_combined_acc`
> (this doc's legacy number) side-by-side. Treat tables in §4 and §5
> below as the *contaminated* numbers until the corrected sweep lands.

---

## Headline

| | Stage A baseline | Stage B λ=0.1 | Stage B λ=0.3 | Stage B λ=1.0 |
|---|---|---|---|---|
| Utility (Acc@1, 48k) | **98.03%** | 96.75% | 94.00% | 84.08% |
| Honest leakage (Stage C, top-1 over 979) | **35.0%** | 35.3% | 35.7% | 29.0% |
| Refreshed utility (after 2-ep refresh) | — | 97.79% | 97.00% | 84.49% |
| Refreshed leakage (Stage C on refreshed) | — | 35.3% | 35.7% | **35.0%** |

**Key finding**: at λ=1.0, Stage B's apparent privacy gain (35→29%) **collapses entirely** under 2 epochs of utility recalibration — leakage rebounds to 35.0%, fully back to baseline, while utility recovers only +0.4 pp. The trade-off Stage B-1.0 nominally made (−14 pp utility, −6 pp leakage) is, post-refresh, "−14 pp utility, **0 pp** privacy." Privacy gain at λ ∈ {0.1, 0.3} was already at noise level pre-refresh.

---

## 1. Experimental setup

| | |
|---|---|
| Hardware | 1× NVIDIA L40 (46 GB), CUDA 12.4 |
| Software | torch 2.6+cu124, torchvision 0.21, sklearn 1.8 |
| Backbone | `resnet18` (pretrained on full ImageNet), `gt=True`, no phase |
| Algorithm | ITIT (`--GT-alg 1`, `--background-K 0`) |
| Channel noise | None (`--SNR` omitted) |
| Privacy loss | `--priv-loss entropy` (`-H(softmax)`) |
| Training data | `data/GroupTestingDataset` (3 firearm + 976 background classes; 21 banned classes excluded by `data_scripts/create_dataset_from_imagenet.py`) |
| Adversary | `AdversaryHead(arch_name='resnet18', num_classes=979)` (979 = total wnids in the reorganized dataset; **not** canonical ImageNet 1000) |
| Stage A epochs | 20, batch 32, LR 1e-3 |
| Stage B epochs | 30, batch 32, LR 1e-3 (encoder), LR 1e-3 (adversary), `--k-adv 5`, `--recovery-epochs 0` |
| Stage C epochs | 30, batch 32, LR 1e-3, fresh-init adversary trained from scratch |
| Refresh epochs | 2, batch 32, LR **1e-4** (E + R unfrozen, on firearm CE only) |
| Random seed | Default (no `--seed`) — single run per config, no variance bands |

**Two distinct val protocols are used in the writeup:**

- **48k protocol** (the one main.py uses, "apples-to-apples" eval): 48,850 samples = 50 firearm files specified by `Constants.firearm_file_paths` + 48,800 background val images. Highly imbalanced. Reports Acc@1 / ROC-AUC / per-class precision-recall.
- **300-sample balanced protocol** (the one `privacy/train_privacy.py` uses for its per-epoch validation): 150 firearm + 150 sampled-background = 300 balanced samples. Noisy with ±2 pp run-to-run, and not directly comparable to the 48k numbers.

Unless explicitly noted, all utility numbers in this document are on the **48k protocol** for comparability.

---

## 2. Stage A — utility pretrain (no privacy)

**Command:**
```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u main.py \
    --data data/GroupTestingDataset \
    --task-num 2 --background-K 0 --GT-alg 1 \
    -a resnet18 --pretrained \
    --epochs 20 --batch-size 32 --lr 0.001 \
    --output_dir Trained_Models/StageA_ITIT_ResNet18 \
    --log-name stage_a.log
```

**Result (model_best.pth.tar, epoch 17, 48k val):**

| Metric | Value |
|---|---|
| Acc@1 | 98.03% (final-epoch checkpoint: 96.86%) |
| ROC-AUC | 1.000 |
| Recall (firearm) | 50/50 = 1.00 |
| Precision (firearm) | 50/(50+964) ≈ 0.05 |
| Confusion matrix | `[[47836, 964], [0, 50]]` |

Wallclock: ~20 min. Receiver classifies firearm vs background perfectly in score-ranking sense (AUC 1.000). The low precision is just the threshold being permissive — no firearms are missed.

---

## 3. Stage B — privacy fine-tune (λ ∈ {0.1, 0.3, 1.0})

**Command (with `--lambda $LAM`):**
```bash
.venv/bin/python -u -m privacy.train_privacy \
    --stage-a-ckpt Trained_Models/StageA_ITIT_ResNet18/model_best.pth.tar \
    --data data/GroupTestingDataset --task-num 2 --background-K 0 --GT-alg 1 \
    -a resnet18 --priv-loss entropy --lambda $LAM --k-adv 5 \
    --stage-b-epochs 30 --recovery-epochs 0 \
    --batch-size 32 -j 8 -valj 4 \
    --output_dir Trained_Models/StageB_ITIT_ResNet18_lam${LAM}_entropy
```

### Per-epoch utility curve on 300-sample balanced val (`val_firearm_acc`)

Selected epochs:

| Epoch | λ=0.1 | λ=0.3 | λ=1.0 |
|---|---|---|---|
| 0 (just after Stage A) | ~0.95 | ~0.94 | 0.93 |
| 10 | ~0.95 | ~0.93 | 0.86 |
| 20 | ~0.95 | ~0.93 | 0.83 |
| 29 (final) | 0.94 | 0.94 | 0.80 |

### Final loss components (Stage B epoch 29)

For 979-way adversary, chance CE = `log(979) ≈ 6.886`. `priv_loss = -H(softmax)` where minimum is `-log(979) ≈ -6.886` (uniform = max privacy).

| | λ=0.1 | λ=0.3 | λ=1.0 |
|---|---|---|---|
| util_loss | ~0.18 | ~0.23 | ~0.42 |
| priv_loss | ~−2.05 | ~−2.10 | ~−3.0 |
| adv_loss | ~1.5 | ~1.8 | ~2.4 |

Reading the loss curves:

- adv_loss (lower = adversary wins) drops steadily across all three λ values — adversary inner loop converges to a strong attacker on the in-loop features.
- priv_loss for λ=1.0 drifts UP from ~−5.4 (early) to ~−3.0 (late), meaning adversary's posterior entropy *decreases* through training. Encoder is **failing** to push the adversary toward uniform. At λ=0.1/0.3 the privacy term is too weak to pull the encoder against the utility loss at all.

### Apples-to-apples utility on 48k (Stage B `stage_b_final.pth.tar`)

Stage B checkpoints were converted to main.py format (key rename `state_dict_backbone` → `state_dict`, prepend `module.` prefix) and evaluated with `main.py --evaluate`:

| λ | Acc@1 | ROC-AUC | Recall (firearm) | False-pos count |
|---|---|---|---|---|
| 0 (Stage A) | 98.03% | 1.000 | 50/50 | 964 |
| 0.1 | 96.75% | 0.998 | 49/50 | 1587 |
| 0.3 | 94.00% | 0.997 | 49/50 | 2931 |
| 1.0 | **84.08%** | **0.950** | **46/50** | 7774 |

The clean monotone pattern: as λ increases, Acc and AUC both drop, false-positive count grows ~8× (964 → 7774), and at λ=1.0 the system actually **misses 4 of the 50 firearms** — a clinically-meaningful regression for the deployed task.

---

## 4. Stage C — honest leakage measurement (fresh-init adversary)

**Command (with `--stage-b-ckpt $CKPT`):**
```bash
.venv/bin/python -u -m privacy.eval_privacy \
    --stage-b-ckpt $CKPT \
    --data data/GroupTestingDataset --task-num 2 --background-K 0 --GT-alg 1 \
    -a resnet18 --stage-c-epochs 30 --batch-size 32 \
    --output_dir <out>
```

A fresh adversary head was trained from scratch (Kaiming init) for 30 epochs on intercepted features. The encoder is fully frozen (`requires_grad=False` on every param + eval mode) so BN running stats and weights don't drift during Stage C.

**Reported metric**: top-1 accuracy of the fresh adversary, computed on the 300-sample balanced val (150 firearm + 150 background) — *not* the 48k val set; that 48k figure refers to `main.py`'s utility-eval protocol and was incorrectly transcribed here. Random chance = 1/979 ≈ 0.1%.

> **Note (2026-04-28):** the 35% values below are dominated by firearm-wnid
> classification accuracy (not background leakage). Under the firearm-filtered
> metric, `top1_background_acc` is the threat-model leakage number; see the
> superseded notice at the top of this doc.

| Stage B ckpt | Top-1 ImageNet leakage |
|---|---|
| Stage A baseline (no privacy) | **35.0%** |
| Stage B λ=0.1 | 35.3% |
| Stage B λ=0.3 | 35.7% |
| Stage B λ=1.0 | **29.0%** |

Reading:

- 35% is **350× chance**. The unprotected encoder leaks a huge amount of fine-grained class info — ResNet-18 layer2 features are very class-discriminative.
- λ ∈ {0.1, 0.3}: leakage statistically indistinguishable from baseline (±0.7 pp around 35%, well within Stage C's run-to-run variance for 30 epochs of fresh-adversary training). **No measurable privacy gain.**
- λ=1.0: real and significant drop, −6 pp absolute, −17% relative. But absolute leakage is still 290× chance.

---

## 5. Refresh + Stage C — does the privacy gain survive recalibration?

The fragility test: take each Stage B checkpoint, do a **brief utility refresh** (E + R both unfrozen, 2 epochs at LR 1e-4 on the binary firearm CE), then re-run Stage C with a fresh adversary on the refreshed model.

**Refresh command:**
```bash
.venv/bin/python -u main.py \
    --resume <stageB_main_format.pth.tar> \
    --data data/GroupTestingDataset \
    --task-num 2 --background-K 0 --GT-alg 1 \
    -a resnet18 --epochs 2 --batch-size 32 --lr 0.0001 \
    --output_dir Trained_Models/RefreshedB_lam${LAM}
```

(Stage B → main.py format conversion is done inline; see `/tmp/refresh_sweep.sh` for the full pipeline.)

### Combined results

| λ | Stage B util (48k) | **Refreshed util** | Stage B leakage | **Refreshed leakage** | Δ leakage (refresh) |
|---|---|---|---|---|---|
| 0 (baseline) | 98.03% | — | 35.0% | — | — |
| 0.1 | 96.75% | **97.79%** | 35.3% | 35.3% | 0.0 pp |
| 0.3 | 94.00% | **97.00%** | 35.7% | 35.7% | 0.0 pp |
| 1.0 | 84.08% | **84.49%** | **29.0%** | **35.0%** | **+6.0 pp** |

### Reading

**For λ ∈ {0.1, 0.3}**: the refresh recovers ~97% utility (essentially back to Stage A baseline) without changing leakage. So the brief recalibration gives back the small utility loss while the (statistically-zero) privacy gain stays at zero. These regimes are best understood as "Stage B made small perturbations that the refresh undoes; nothing was ever truly removed."

**For λ=1.0**: this is the striking result.

- Utility: 84.08% → 84.49% (+0.4 pp). The encoder is in a state from which 2 epochs of E+R refresh cannot recover the firearm classification capability.
- Leakage: 29.0% → 35.0% (+6.0 pp). **The entire Stage B privacy gain is undone** — leakage matches the unprotected baseline exactly.

So the trade Stage B nominally made — pay 14 pp utility, get 6 pp privacy — is, post-refresh, **"pay 14 pp utility, get 0 pp privacy."** The privacy was an artifact of the receiver having had no opportunity to relearn the encoder's new feature representation. Once the receiver updates (along with the encoder, in a 2-epoch refresh), the adversary trained on the refreshed features extracts the same 35% as before.

### What this means structurally

This validates the **scrambling failure mode** that motivated the frozen-receiver design — and reveals that the frozen-receiver constraint, in its current form, is **not a complete fix**:

- The frozen receiver did prevent the encoder from completely scrambling features (utility dropped only to 84%, not to chance — encoder *did* stay mostly within R's preimage).
- But the encoder also found a feature subspace where the firearm-relevant projection is partially garbled (utility −14 pp) while the ImageNet-discriminative information **is still present** in the features (leakage rebounds fully).
- A worst-case eavesdropper who can also brief-recalibrate (e.g., they capture features over time, build their own labeled corpus, fine-tune their attack model) extracts the full 35% leakage.
- **The honest privacy number for λ=1.0 is therefore 35%, not 29%.** Reporting 29% in a paper would overstate the privacy guarantee.

---

## 6. Caveats

1. **Single seed, single run per config.** No variance bands. Stage C in particular has run-to-run noise of ±1 pp at 30 epochs; smaller deltas are noise.
2. **Adversary task is 979-way, not canonical 1000-way.** The 21 "banned" classes from `create_dataset_from_imagenet.py` are absent from training and val. Comparisons across configs are valid; comparisons to externally-published "1000-way ImageNet top-1" numbers are not.
3. **Stage C trained for only 30 epochs** (not the spec's default 60). A longer Stage C might find more leakage — these numbers should be considered lower bounds on the worst-case attacker.
4. **Refresh hyperparameters are arbitrary.** 2 epochs at LR 1e-4 is one choice. A larger-LR / longer refresh would likely recover more utility and possibly more leakage; a smaller refresh might leave both signals where they were. The qualitative finding (privacy at λ=1.0 collapses easily) is robust, but the exact crossover point is not characterized.
5. **The 300-sample balanced val set used by `validate_utility` inside Stage B is noisy** (±2 pp). Per-epoch utility curves should be read for trend, not point values.
6. **ResNet-18 + ITIT is a specific (and probably worst-case) setting.** The frozen-receiver-defeats-scrambling argument may hold better with bigger backbones (more representational capacity in the receiver) and/or with GTGT-FM (channel-sum mixing provides structural privacy that an attacker cannot unscramble).
7. **Utility refresh updates BOTH E and R.** A cleaner experiment would be receiver-only refresh (E frozen) — that would isolate "does the receiver alone re-learn the scrambled code" from "does the encoder drift back toward leakage during refresh." Not run in this batch.
8. **The gradient-flow contract has been verified** by the unit tests in `tests/test_trainer.py` (BN running stats stay frozen during Stage B; adversary params don't accumulate gradient in the encoder outer step). The fragility observed here is *not* due to a leak in those contracts — it's a property of the trained encoder's information content.

---

## 7. Tentative interpretation and open questions

**Tentative interpretation**: For ITIT with a ResNet-18-sized layer2 bottleneck, the encoder has too few degrees of freedom to find features that simultaneously (a) keep the receiver's firearm-classification path intact and (b) genuinely strip ImageNet information. Within the receiver's preimage, the encoder can make features that *look* less attackable to an in-loop adversary (privacy term is locally satisfied) but the actual MI with class label is preserved — discoverable by any attacker that retrains. The frozen-receiver constraint is a structural anti-scrambling measure, but it only prevents *complete* scrambling, not partial reparametrizations that still preserve information.

**Open questions:**

1. Does a **bigger backbone** (ResNet-50, ResNeXt-101) genuinely change the picture, or does it just push the same fragility curve to a different operating point? Larger receivers have more decoding capacity, so the encoder's feasible feature manifold is wider — possibly enough to find true low-MI representations.
2. Does **GTGT-FM K=8** give us robust privacy "for free" (via the channel sum), or does the same fragility pattern emerge there? This is probably the most informative single experiment to run next.
3. Does the **CE form** (`-F.cross_entropy(adv_logits, target)` instead of `-H`) behave differently under the refresh test? Same scrambling failure expected, but worth checking.
4. Would a **longer refresh** (10 epochs vs 2) recover utility at λ=1.0 too — i.e., is the encoder "stuck" or just slow? If utility comes back fully, then we've confirmed Stage B at λ=1.0 was net-zero (utility AND privacy returned to baseline).
5. Would **receiver-only refresh** (E frozen, R re-trained on utility) tell a different story than the current E+R refresh?
6. What's the maximum λ before the encoder collapses entirely (e.g., features become noise, both metrics tank)? Knowing the structural ceiling characterizes the algorithm's fundamental capability.

---

## 8. File pointers

All checkpoints are gitignored under `Trained_Models/`. Stdout logs (`*_stdout.log` files) are also gitignored.

| Artifact | Path |
|---|---|
| Stage A best ckpt | `Trained_Models/StageA_ITIT_ResNet18/model_best.pth.tar` |
| Stage A stdout | `Trained_Models/StageA_ITIT_ResNet18_stdout.log` |
| Stage B λ=0.1 ckpt | `Trained_Models/StageB_ITIT_ResNet18_lam0.1_entropy/stage_b_final.pth.tar` |
| Stage B λ=0.3 ckpt | `Trained_Models/StageB_ITIT_ResNet18_lam0.3_entropy/stage_b_final.pth.tar` |
| Stage B λ=1.0 ckpt | `Trained_Models/StageB_ITIT_ResNet18_lam1.0_entropy/stage_b_final.pth.tar` |
| Stage C on Stage A leakage | `Trained_Models/StageC_OnStageA_ITIT_ResNet18/leakage.json` |
| Stage C on Stage B λ=0.1 leakage | `Trained_Models/StageC_OnStageB_ITIT_ResNet18_lam0.1_entropy/leakage.json` |
| Stage C on Stage B λ=0.3 leakage | `Trained_Models/StageC_OnStageB_ITIT_ResNet18_lam0.3_entropy/leakage.json` |
| Stage C on Stage B λ=1.0 leakage | `Trained_Models/StageC_OnStageB_ITIT_ResNet18_lam1.0_entropy/leakage.json` |
| Refreshed ckpt λ=0.1 | `Trained_Models/RefreshedB_lam0.1/checkpoint.pth.tar` |
| Refreshed ckpt λ=0.3 | `Trained_Models/RefreshedB_lam0.3/checkpoint.pth.tar` |
| Refreshed ckpt λ=1.0 | `Trained_Models/RefreshedB_lam1.0/checkpoint.pth.tar` |
| Stage C on refreshed λ=0.1 | `Trained_Models/StageC_OnRefreshedB_lam0.1/leakage.json` |
| Stage C on refreshed λ=0.3 | `Trained_Models/StageC_OnRefreshedB_lam0.3/leakage.json` |
| Stage C on refreshed λ=1.0 | `Trained_Models/StageC_OnRefreshedB_lam1.0/leakage.json` |
| Sweep scripts (transient) | `/tmp/lambda_sweep.sh`, `/tmp/refresh_sweep.sh` |

---

## 9. Suggested next experiments (ranked)

1. **GTGT-FM K=8, ResNet-18** with the same λ sweep + refresh-then-Stage-C protocol. The channel sum is a structural mixing operation that's hard for an attacker to invert; the resulting privacy should be much more robust to refresh. Cheap (~1 hour for the full sweep including refresh tests).
2. **ResNeXt-101_32x8d, ITIT, no noise**, λ sweep + refresh tests. Tests whether bigger backbone capacity changes the fragility picture. Multi-hour (Stage A pretrain alone is several hours).
3. **CE-form privacy loss**: same ITIT/ResNet-18 setup, just `--priv-loss ce` instead of entropy. Probably same fragility, but cheap to confirm and rules out the loss form as the cause.
4. **Receiver-only refresh** to isolate "is leakage rebound from receiver re-learning, or from encoder drift during refresh?". One-line script change to `_freeze_encoder` during the refresh.
5. **Longer refresh** at λ=1.0 (10 epochs instead of 2) to see if utility eventually recovers — or if the encoder is in a permanent bad spot.
