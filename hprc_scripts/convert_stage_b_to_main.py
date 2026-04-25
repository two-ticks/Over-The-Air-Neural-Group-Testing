"""Convert a privacy/train_privacy.py Stage B checkpoint to main.py's resume format.

main.py --resume expects:
  state_dict     : with 'module.' prefix (DataParallel)
  optimizer      : optional (re-init if missing)
  epoch          : optional
  arch           : optional
  best_acc1      : optional
  coded_pwr      : optional

Stage B writes:
  state_dict_backbone     : without 'module.' prefix
  state_dict_adversary    : (we drop this for refresh)
  args                    : (we drop)
  coded_pwr

This script just renames the key and re-adds the 'module.' prefix so
main.py --resume happily loads it. Adversary state and Stage B args are
deliberately discarded — refresh is utility-only.

Used by hprc_scripts/ce_sweep.slurm.

Usage:
    python hprc_scripts/convert_stage_b_to_main.py --in <stage_b_final.pth.tar> --out <main_fmt.pth.tar>
"""
import argparse
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="src", required=True, help="path to stage_b_final.pth.tar")
    p.add_argument("--out", dest="dst", required=True, help="output path for main.py-format ckpt")
    args = p.parse_args()

    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    if "state_dict_backbone" not in ckpt:
        raise SystemExit(
            f"{args.src} doesn't have a 'state_dict_backbone' key — is this really a Stage B checkpoint?"
        )

    bb = ckpt["state_dict_backbone"]
    bb = {f"module.{k}": v for k, v in bb.items()}

    out = {
        "state_dict": bb,
        "epoch": 0,
        "arch": ckpt.get("args", {}).get("arch", "resnet18"),
        "best_acc1": 0.0,
        "coded_pwr": float(ckpt.get("coded_pwr", 1.0)),
    }
    torch.save(out, args.dst)
    print(f"Wrote {args.dst}")
    print(f"  state_dict keys: {len(bb)}")
    print(f"  coded_pwr: {out['coded_pwr']}")


if __name__ == "__main__":
    main()
