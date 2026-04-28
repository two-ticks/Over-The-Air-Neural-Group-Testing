"""Stage C: train a fresh-init adversary from scratch on a frozen privacy-fine-tuned encoder.

Reports its top-1 ImageNet accuracy (ITIT) or per-class mean ROC-AUC + mAP (GTGT-FM)
as the honest leakage measurement.

Usage:
    python -m privacy.eval_privacy \
        --stage-b-ckpt Trained_Models/PrivacySmoke/stage_b_final.pth.tar \
        --data data/GroupTestingDataset --task-num 2 --background-K 0 \
        --GT-alg 1 -a resnet18 --stage-c-epochs 5 --batch-size 16 \
        --output_dir Trained_Models/PrivacySmoke/EvalC
"""
import argparse
import json
import os
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import resnet_design2 as models
from privacy.adversary import AdversaryHead
from privacy.dataset import PrivacyTaskCoalitionDataset
from privacy._snr import snr_to_noise_std


def get_parser():
    p = argparse.ArgumentParser("Privacy-preserving OTA-NGT — Stage C honest leakage eval")
    p.add_argument("--data", required=True)
    p.add_argument("--task-num", type=int, default=2)
    p.add_argument("--background-K", type=int, required=True)
    p.add_argument("--GT-alg", type=int, choices=[1, 2], required=True)
    p.add_argument("-a", "--arch", required=True)
    p.add_argument("--phase", action="store_true")
    p.add_argument("--SNR", type=float, default=None)
    p.add_argument("--stage-b-ckpt", required=True)
    p.add_argument("--stage-c-epochs", type=int, default=60)
    p.add_argument("--adv-lr", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("-j", "--workers", type=int, default=8)
    p.add_argument("-valj", "--val-workers", type=int, default=4)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--log-name", default="eval_c.log")
    p.add_argument("--print-freq", type=int, default=50)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--save-adversary", action="store_true", default=True,
                   help="Persist trained Stage C adversary so future re-evals are free.")
    p.add_argument("--no-save-adversary", dest="save_adversary", action="store_false")
    return p


def build_datasets(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_t = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_t = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    train_list, val_list = [], []
    for folder_idx in range(args.task_num):
        train_list.append(datasets.ImageFolder(os.path.join(args.data, str(folder_idx), "train"), train_t))
        val_list.append(datasets.ImageFolder(os.path.join(args.data, str(folder_idx), "val"), val_t))
    return train_list, val_list


def load_backbone(args, device):
    ctor = getattr(models, args.arch)
    backbone = ctor(pretrained=False, gt=True, phase=args.phase)
    ckpt = torch.load(args.stage_b_ckpt, map_location="cpu", weights_only=False)
    coded_pwr = float(ckpt.get("coded_pwr", 1.0))
    state = ckpt["state_dict_backbone"] if "state_dict_backbone" in ckpt else ckpt["state_dict"]
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    backbone.load_state_dict(state)
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone, coded_pwr


def _build_background_mask(firearm_target: torch.Tensor, K: int) -> torch.Tensor:
    """Per-stacked-image background mask for ITIT.

    Stacked-image layout per item: slot 0 is the mixing slot (firearm if
    firearm_target==1, else background); slots 1..K-1 are always backgrounds
    sampled from the negative pool. So slot 0 is firearm iff firearm_target==1;
    every other slot is always a background.

    Returns a (B, K) bool tensor; True means "this stacked image is a
    background sample" (i.e., belongs to the threat-model adversary's
    label space)."""
    B = firearm_target.shape[0]
    mask = torch.ones(B, K, dtype=torch.bool, device=firearm_target.device)
    mask[:, 0] = firearm_target == 0
    return mask


def train_fresh_adversary(backbone, adversary, train_dataset, args, device, log, coded_pwr=1.0):
    snr_noise = snr_to_noise_std(args.SNR, args.GT_alg, coded_pwr)
    optim = torch.optim.SGD(adversary.parameters(), lr=args.adv_lr, momentum=args.momentum,
                            weight_decay=args.weight_decay)
    for epoch in range(args.stage_c_epochs):
        loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=True,
        )
        adversary.train()
        t0 = time.time()
        for it, (images, firearm_target, imagenet_target_per_image) in enumerate(loader):
            images = images.to(device); imagenet_target_per_image = imagenet_target_per_image.to(device)
            firearm_target = firearm_target.to(device)
            with torch.no_grad():
                pre = backbone.encode(images)
                post, _, _ = backbone.channel(pre, noise_std=snr_noise,
                                              gpu=device.index if device.type == "cuda" else None)
            if args.GT_alg == 1:
                B, K, Cf, Hf, Wf = pre.shape
                adv_in = pre.reshape(B * K, Cf, Hf, Wf)
                adv_target = imagenet_target_per_image.reshape(-1)
                bg_mask_flat = _build_background_mask(firearm_target, K).reshape(-1)
                if not bg_mask_flat.any():
                    continue
                adv_in = adv_in[bg_mask_flat]
                adv_target = adv_target[bg_mask_flat]
                logits = adversary(adv_in)
                loss = F.cross_entropy(logits, adv_target)
            else:
                num_classes = adversary.fc.out_features
                bg_item_mask = firearm_target == 0
                if not bg_item_mask.any():
                    continue
                post_bg = post[bg_item_mask]
                imagenet_targets_bg = imagenet_target_per_image[bg_item_mask]
                khot = torch.zeros(post_bg.size(0), num_classes, device=device).scatter_(
                    1, imagenet_targets_bg, 1.0)
                logits = adversary(post_bg)
                loss = F.binary_cross_entropy_with_logits(logits, khot)
            optim.zero_grad(set_to_none=True); loss.backward(); optim.step()
            if it % args.print_freq == 0:
                line = f"[StageC][ep {epoch}][it {it:5d}] adv_loss={loss.item():.4f}"
                print(line); log.write(line + "\n"); log.flush()
        line = f"[StageC][ep {epoch}] time={time.time()-t0:.1f}s"
        print(line); log.write(line + "\n"); log.flush()


def evaluate_leakage(backbone, adversary, val_dataset, args, device, coded_pwr=1.0):
    """Returns dict with leakage metrics on val set.

    Threat model: only background (non-firearm) samples count as leakage.
    The firearm-vs-background classification is the receiver's legitimate,
    intentionally-exposed output. We track three numbers for ITIT so the
    contamination is auditable:

    * `top1_background_acc` — the threat-model leakage metric.
    * `top1_firearm_acc`    — adversary accuracy on firearm samples; should
                              be high regardless of privacy training; informational only.
    * `top1_combined_acc`   — legacy number (matches the pre-fix `top1_imagenet_acc`).
    """
    snr_noise = snr_to_noise_std(args.SNR, args.GT_alg, coded_pwr)
    loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.val_workers, pin_memory=True, drop_last=False,
    )
    adversary.eval()
    correct_bg = total_bg = 0
    correct_fa = total_fa = 0
    all_logits, all_targets, all_bg_item_masks = [], [], []
    with torch.no_grad():
        for images, firearm_target, imagenet_target_per_image in loader:
            images = images.to(device); imagenet_target_per_image = imagenet_target_per_image.to(device)
            firearm_target = firearm_target.to(device)
            pre = backbone.encode(images)
            post, _, _ = backbone.channel(pre, noise_std=snr_noise,
                                          gpu=device.index if device.type == "cuda" else None)
            if args.GT_alg == 1:
                B, K, Cf, Hf, Wf = pre.shape
                adv_in = pre.reshape(B * K, Cf, Hf, Wf)
                adv_target = imagenet_target_per_image.reshape(-1)
                bg_mask_flat = _build_background_mask(firearm_target, K).reshape(-1)
                fa_mask_flat = ~bg_mask_flat
                logits = adversary(adv_in)
                pred = logits.argmax(dim=-1)
                hits = (pred == adv_target)
                correct_bg += (hits & bg_mask_flat).sum().item()
                total_bg += bg_mask_flat.sum().item()
                correct_fa += (hits & fa_mask_flat).sum().item()
                total_fa += fa_mask_flat.sum().item()
            else:
                logits = adversary(post)
                all_logits.append(logits.cpu().numpy())
                num_classes = adversary.fc.out_features
                khot = torch.zeros(images.size(0), num_classes, device=device).scatter_(
                    1, imagenet_target_per_image, 1.0)
                all_targets.append(khot.cpu().numpy())
                all_bg_item_masks.append((firearm_target == 0).cpu().numpy())

    if args.GT_alg == 1:
        total_combined = total_bg + total_fa
        correct_combined = correct_bg + correct_fa
        return {
            "top1_background_acc": correct_bg / max(total_bg, 1),
            "top1_firearm_acc": correct_fa / max(total_fa, 1),
            "top1_combined_acc": correct_combined / max(total_combined, 1),
            "n_background_eval": total_bg,
            "n_firearm_eval": total_fa,
            "n_combined_eval": total_combined,
            # Legacy key — same value as top1_combined_acc, kept so existing
            # downstream tooling doesn't break.
            "top1_imagenet_acc": correct_combined / max(total_combined, 1),
        }

    # GTGT-FM: per-class AUC + mAP. Compute combined (legacy) and background-only.
    from sklearn.metrics import roc_auc_score, average_precision_score
    logits = np.concatenate(all_logits, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    bg_item_mask = np.concatenate(all_bg_item_masks, axis=0).astype(bool)

    def _per_class_aucs(logits_arr, targets_arr):
        aucs, aps = [], []
        for k in range(targets_arr.shape[1]):
            if targets_arr[:, k].sum() == 0:
                continue
            aucs.append(roc_auc_score(targets_arr[:, k], logits_arr[:, k]))
            aps.append(average_precision_score(targets_arr[:, k], logits_arr[:, k]))
        return aucs, aps

    aucs_combined, aps_combined = _per_class_aucs(logits, targets)
    aucs_bg, aps_bg = _per_class_aucs(logits[bg_item_mask], targets[bg_item_mask])
    return {
        "mean_auc_background": float(np.mean(aucs_bg)) if aucs_bg else float("nan"),
        "mean_ap_background": float(np.mean(aps_bg)) if aps_bg else float("nan"),
        "n_classes_evaluated_background": len(aucs_bg),
        "n_background_items": int(bg_item_mask.sum()),
        "mean_auc_combined": float(np.mean(aucs_combined)) if aucs_combined else float("nan"),
        "mean_ap_combined": float(np.mean(aps_combined)) if aps_combined else float("nan"),
        "n_classes_evaluated_combined": len(aucs_combined),
        # Legacy keys.
        "mean_auc": float(np.mean(aucs_combined)) if aucs_combined else float("nan"),
        "mean_ap": float(np.mean(aps_combined)) if aps_combined else float("nan"),
        "num_classes_evaluated": len(aucs_combined),
    }


def main():
    args = get_parser().parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log = open(os.path.join(args.output_dir, args.log_name), "w")
    log.write(f"args: {json.dumps(vars(args), indent=2)}\n"); log.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, coded_pwr = load_backbone(args, device)
    train_list, val_list = build_datasets(args)

    # Threat-model mapping: only background wnids are in the adversary's label space.
    # Firearm (folder 0) is the receiver's intentionally-exposed output; an attacker
    # learning "this is a firearm" is by design, not a privacy violation. The Stage C
    # adversary must therefore never see firearm samples and never have firearm wnids
    # as candidate predictions.
    background_train_list = train_list[1:]
    background_val_list = val_list[1:]
    bg_wnids = sorted(set().union(
        *(ds.class_to_idx.keys() for ds in background_train_list + background_val_list)
    ))
    wnid_to_imagenet_idx = {w: i for i, w in enumerate(bg_wnids)}

    train_dataset = PrivacyTaskCoalitionDataset(train_list, args, split="train",
                                                wnid_to_imagenet_idx=wnid_to_imagenet_idx,
                                                background_only=True)
    val_dataset = PrivacyTaskCoalitionDataset(val_list, args, split="val",
                                              wnid_to_imagenet_idx=wnid_to_imagenet_idx,
                                              background_only=True)

    adversary = AdversaryHead(arch_name=args.arch, num_classes=train_dataset.num_imagenet_classes).to(device)

    train_fresh_adversary(backbone, adversary, train_dataset, args, device, log, coded_pwr=coded_pwr)

    metrics = evaluate_leakage(backbone, adversary, val_dataset, args, device, coded_pwr=coded_pwr)
    print("Leakage metrics:", json.dumps(metrics, indent=2))
    log.write("leakage: " + json.dumps(metrics) + "\n"); log.flush()
    with open(os.path.join(args.output_dir, "leakage.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if args.save_adversary:
        ckpt_path = os.path.join(args.output_dir, "adversary.pth.tar")
        torch.save({
            "state_dict_adversary": adversary.state_dict(),
            "arch_name": args.arch,
            "num_classes": train_dataset.num_imagenet_classes,
            "GT_alg": args.GT_alg,
            "stage_b_ckpt": args.stage_b_ckpt,
        }, ckpt_path)
        log.write(f"saved adversary to {ckpt_path}\n"); log.flush()
    log.close()


if __name__ == "__main__":
    main()
