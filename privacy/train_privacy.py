"""Stage B + Stage A_recovery entry point for privacy-preserving OTA-NGT training.

Usage:
    python -m privacy.train_privacy \
        --stage-a-ckpt Trained_Models/SmokeTest/checkpoint.pth.tar \
        --data data/GroupTestingDataset --task-num 2 --background-K 0 \
        --GT-alg 1 -a resnet18 --priv-loss entropy --lambda 1.0 --k-adv 5 \
        --stage-b-epochs 5 --recovery-epochs 1 --batch-size 16 \
        --output_dir Trained_Models/PrivacySmoke
"""
import argparse
import json
import os
import pathlib
import sys
import time

import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import resnet_design2 as models
from privacy.adversary import AdversaryHead
from privacy.dataset import PrivacyTaskCoalitionDataset
from privacy.trainer import stage_b_step
from privacy._snr import snr_to_noise_std


def get_parser():
    p = argparse.ArgumentParser("Privacy-preserving OTA-NGT — Stage B + recovery")
    # Core data + arch (mirror main.py where applicable)
    p.add_argument("--data", required=True, help="path to GroupTestingDataset root")
    p.add_argument("--task-num", type=int, default=2)
    p.add_argument("--background-K", type=int, required=True, help="group_size - 1; 0 for ITIT")
    p.add_argument("--GT-alg", type=int, choices=[1, 2], required=True, help="1=ITIT, 2=GTGT-FM")
    p.add_argument("-a", "--arch", required=True, help="model arch in resnet_design2")
    p.add_argument("--phase", action="store_true", help="enable random phase shift channel")
    # Channel / SNR (Stage B inherits Stage A's channel)
    p.add_argument("--SNR", type=float, default=None)
    # Stage A checkpoint
    p.add_argument("--stage-a-ckpt", required=True, help="path to checkpoint produced by main.py")
    # Stage B knobs
    p.add_argument("--priv-loss", choices=["ce", "entropy"], default="entropy")
    p.add_argument("--lambda", type=float, default=1.0, dest="lam")
    p.add_argument("--k-adv", type=int, default=5)
    p.add_argument("--stage-b-epochs", type=int, default=30)
    p.add_argument("--recovery-epochs", type=int, default=2)
    # Optimizers
    p.add_argument("--enc-lr", type=float, default=1e-3)
    p.add_argument("--adv-lr", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    # Loader
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("-j", "--workers", type=int, default=8)
    p.add_argument("-valj", "--val-workers", type=int, default=4)
    # Output
    p.add_argument("--output_dir", required=True)
    p.add_argument("--log-name", default="privacy.log")
    p.add_argument("--print-freq", type=int, default=20)
    p.add_argument("--seed", type=int, default=None)
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
        train_dir = os.path.join(args.data, str(folder_idx), "train")
        val_dir = os.path.join(args.data, str(folder_idx), "val")
        train_list.append(datasets.ImageFolder(train_dir, train_t))
        val_list.append(datasets.ImageFolder(val_dir, val_t))
    return train_list, val_list


def load_stage_a(args, device):
    """Build a fresh `ResNet_GT` (matching args), load Stage-A weights, return it and coded_pwr."""
    ctor = getattr(models, args.arch)
    model = ctor(pretrained=False, gt=True, phase=args.phase)
    ckpt = torch.load(args.stage_a_ckpt, map_location="cpu", weights_only=False)
    coded_pwr = float(ckpt.get("coded_pwr", 1.0))
    state = ckpt["state_dict"]
    # Strip "module." prefix if Stage A used DataParallel.
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(device), coded_pwr


def validate_utility(backbone, val_dataset, args, device, coded_pwr=1.0):
    """Quick utility validation: firearm Acc@1 on a single-pass val loader."""
    backbone.eval()
    loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.val_workers, pin_memory=True, drop_last=False,
    )
    correct = total = 0
    with torch.no_grad():
        for images, firearm_target, _ in loader:
            images = images.to(device)
            firearm_target = firearm_target.to(device)
            pre = backbone.encode(images)
            post, _, _ = backbone.channel(pre,
                                          noise_std=snr_to_noise_std(args.SNR, args.GT_alg, coded_pwr),
                                          gpu=device.index if device.type == "cuda" else None)
            logits = backbone.decode(post)
            pred = logits.argmax(dim=-1)
            correct += (pred == firearm_target).sum().item()
            total += firearm_target.numel()
    return correct / max(total, 1)


def stage_b_loop(backbone, adversary, train_dataset, val_dataset, args, device, log, coded_pwr=1.0):
    enc_params = list(backbone.conv1.parameters()) + list(backbone.bn1.parameters()) \
        + list(backbone.layer1.parameters()) + list(backbone.layer2.parameters())
    enc_optim = torch.optim.SGD(enc_params, lr=args.enc_lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    adv_optim = torch.optim.SGD(adversary.parameters(), lr=args.adv_lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)

    snr_noise = snr_to_noise_std(args.SNR, args.GT_alg, coded_pwr)

    for epoch in range(args.stage_b_epochs):
        loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=True,
        )
        t0 = time.time()
        for it, (images, firearm_target, imagenet_target_per_image) in enumerate(loader):
            # Threat-model mask: only background (non-firearm) items contribute
            # to the encoder's privacy loss; the firearm-vs-background signal is
            # the receiver's legitimate output, not a privacy violation.
            background_mask = (firearm_target == 0)
            metrics = stage_b_step(
                backbone=backbone, adversary=adversary,
                enc_optimizer=enc_optim, adv_optimizer=adv_optim,
                images=images, firearm_target=firearm_target,
                imagenet_target_per_image=imagenet_target_per_image,
                priv_loss_name=args.priv_loss, lam=args.lam, k_adv=args.k_adv,
                device=device, gt_alg=args.GT_alg, background_K=args.background_K,
                snr_noise_std=snr_noise,
                background_mask=background_mask,
            )
            if it % args.print_freq == 0:
                line = (f"[StageB][ep {epoch}][it {it:5d}] "
                        f"util={metrics['loss_util']:.4f} priv={metrics['loss_priv']:.4f} "
                        f"adv={metrics['loss_adv']:.4f} total={metrics['loss_total']:.4f}")
                print(line); log.write(line + "\n"); log.flush()
        acc = validate_utility(backbone, val_dataset, args, device, coded_pwr=coded_pwr)
        line = f"[StageB][ep {epoch}] val_firearm_acc={acc:.4f} time={time.time()-t0:.1f}s"
        print(line); log.write(line + "\n"); log.flush()


def stage_a_recovery_loop(backbone, train_dataset, val_dataset, args, device, log, coded_pwr=1.0):
    """Brief unfreeze-and-train-on-utility-only pass to recover utility regression from Stage B."""
    for p in backbone.parameters():
        p.requires_grad = True
    backbone.train()
    optim = torch.optim.SGD(backbone.parameters(), lr=args.enc_lr, momentum=args.momentum,
                            weight_decay=args.weight_decay)
    snr_noise = snr_to_noise_std(args.SNR, args.GT_alg, coded_pwr)
    for epoch in range(args.recovery_epochs):
        loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=True,
        )
        t0 = time.time()
        for it, (images, firearm_target, _) in enumerate(loader):
            images = images.to(device)
            firearm_target = firearm_target.to(device)
            pre = backbone.encode(images)
            post, _, _ = backbone.channel(pre, noise_std=snr_noise,
                                          gpu=device.index if device.type == "cuda" else None)
            logits = backbone.decode(post)
            loss = F.cross_entropy(logits, firearm_target)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            if it % args.print_freq == 0:
                line = f"[Recovery][ep {epoch}][it {it:5d}] util={loss.item():.4f}"
                print(line); log.write(line + "\n"); log.flush()
        acc = validate_utility(backbone, val_dataset, args, device, coded_pwr=coded_pwr)
        line = f"[Recovery][ep {epoch}] val_firearm_acc={acc:.4f} time={time.time()-t0:.1f}s"
        print(line); log.write(line + "\n"); log.flush()


def save_ckpt(backbone, adversary, args, path, coded_pwr=1.0):
    torch.save({
        "state_dict_backbone": backbone.state_dict(),
        "state_dict_adversary": adversary.state_dict(),
        "args": vars(args),
        "coded_pwr": coded_pwr,
    }, path)


def main():
    args = get_parser().parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(args.output_dir, args.log_name)
    log = open(log_path, "w")
    log.write(f"args: {json.dumps(vars(args), indent=2)}\n"); log.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    backbone, coded_pwr = load_stage_a(args, device)
    train_list, val_list = build_datasets(args)

    # Build a unified wnid mapping from both train and val to avoid label-space divergence (I2).
    all_wnids = sorted(set().union(*(ds.class_to_idx.keys() for ds in train_list + val_list)))
    wnid_to_imagenet_idx = {w: i for i, w in enumerate(all_wnids)}

    train_dataset = PrivacyTaskCoalitionDataset(train_list, args, split="train",
                                                wnid_to_imagenet_idx=wnid_to_imagenet_idx)
    val_dataset = PrivacyTaskCoalitionDataset(val_list, args, split="val",
                                              wnid_to_imagenet_idx=wnid_to_imagenet_idx)

    adversary = AdversaryHead(arch_name=args.arch, num_classes=train_dataset.num_imagenet_classes).to(device)
    print(f"Adversary num_classes={train_dataset.num_imagenet_classes}")

    stage_b_loop(backbone, adversary, train_dataset, val_dataset, args, device, log, coded_pwr=coded_pwr)
    save_ckpt(backbone, adversary, args, os.path.join(args.output_dir, "stage_b_final.pth.tar"),
              coded_pwr=coded_pwr)

    if args.recovery_epochs > 0:
        stage_a_recovery_loop(backbone, train_dataset, val_dataset, args, device, log, coded_pwr=coded_pwr)
        save_ckpt(backbone, adversary, args, os.path.join(args.output_dir, "stage_b_recovered.pth.tar"),
                  coded_pwr=coded_pwr)

    log.close()


if __name__ == "__main__":
    main()
