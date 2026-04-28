"""One-iteration Stage-B step. Pure function-of-modules so it's unit-testable.

The `backbone` is a `ResNet_GT` (or `ResNet_GT_phase`) instance. Stage-B contract:
  * encoder layers (conv1, bn1, layer1, layer2) — TRAIN, gradient flows from utility AND privacy terms
  * receiver layers (layer3, layer4, fc) — FROZEN, no gradient
  * adversary head — TRAIN, separately (inner loop), gradient from CE/BCE on intercepted features
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import priv_loss_ce, priv_loss_entropy, priv_loss_entropy_multilabel


def _freeze_receiver(backbone):
    """Set requires_grad=False on layer3, layer4, fc; eval() on layer3/4 BN running stats."""
    for m in (backbone.layer3, backbone.layer4, backbone.fc):
        for p in m.parameters():
            p.requires_grad = False
        m.eval()  # freeze BN running_mean/running_var/num_batches_tracked


def _flatten_imagenet_target(target_per_image: torch.Tensor) -> torch.Tensor:
    """ITIT (single-label): (B, K+1) -> (B*(K+1),) flat label per stacked image."""
    return target_per_image.reshape(-1)


def _to_khot(target_per_image: torch.Tensor, num_classes: int) -> torch.Tensor:
    """GTGT-FM (multilabel): (B, K+1) -> (B, num_classes) K-hot float tensor."""
    B, _ = target_per_image.shape
    out = torch.zeros(B, num_classes, dtype=torch.float32, device=target_per_image.device)
    out.scatter_(1, target_per_image, 1.0)
    return out


def stage_b_step(
    *,
    backbone,
    adversary,
    enc_optimizer,
    adv_optimizer,
    images: torch.Tensor,                  # (B, K+1, C, H, W)
    firearm_target: torch.Tensor,          # (B,)
    imagenet_target_per_image: torch.Tensor,  # (B, K+1)
    priv_loss_name: str,                   # "ce" or "entropy"
    lam: float,                            # privacy weight
    k_adv: int,                            # adversary inner steps
    device: torch.device,
    gt_alg: int,                           # 1 (ITIT) or 2 (GTGT-FM)
    background_K: int,                     # group_size - 1
    snr_noise_std,                         # float or None
    background_mask: Optional[torch.Tensor] = None,  # (B,) bool; True = "this item is background-only"
):
    """One Stage-B iteration on a minibatch. Returns a metrics dict.

    Side effects:
      * Updates `adversary` parameters via `adv_optimizer` (k_adv steps).
      * Updates encoder params via `enc_optimizer` (1 step).
      * Receiver params are NOT updated (frozen by design).

    `background_mask` (optional, shape `(B,)` bool) implements the threat-model
    fix: when provided, the privacy term in the encoder's outer step is
    computed only over items where the mask is True (i.e., non-firearm items).
    The inner adversary loop is intentionally NOT masked — we want a strong
    attacker that has seen the full feature distribution; we only filter what
    the encoder is penalized for. If None, full batch is used (legacy behavior).
    """
    assert priv_loss_name in ("ce", "entropy"), f"priv_loss_name must be 'ce' or 'entropy', got {priv_loss_name!r}"
    assert gt_alg in (1, 2), f"only ITIT (1) and GTGT-FM (2) are supported in Stage B; got {gt_alg}"

    _freeze_receiver(backbone)
    backbone = backbone.to(device)
    adversary = adversary.to(device)
    images = images.to(device)
    firearm_target = firearm_target.to(device)
    imagenet_target_per_image = imagenet_target_per_image.to(device)
    if background_mask is not None:
        background_mask = background_mask.to(device)

    # --- Adversary inner loop (TTUR): k_adv steps on the same minibatch ---
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False  # freeze backbone temporarily so grads only hit adversary

    # Cache post-channel features once.
    with torch.no_grad():
        pre = backbone.encode(images)
        post_channel, _, _ = backbone.channel(pre, noise_std=snr_noise_std,
                                              gpu=device.index if device.type == "cuda" else None)

    if gt_alg == 1:
        # ITIT: per-image features. Shape after encode: (B, K+1, C', H', W').
        # But channel() collapses K -> 1 (sums and divides). For ITIT, K+1 = 1 typically; if it's >1,
        # we still treat each image's individual feature path. Re-encode without summing for the adversary.
        # Strategy: re-run encode and treat each (B, K+1, C', H', W') slice as a sample.
        pre_for_adv = pre  # (B, K+1, C', H', W')
        B, K, Cf, Hf, Wf = pre_for_adv.shape
        adv_input = pre_for_adv.reshape(B * K, Cf, Hf, Wf)
        adv_target_flat = _flatten_imagenet_target(imagenet_target_per_image)
    else:
        # GTGT-FM: adversary sees the summed post-channel feature.
        adv_input = post_channel  # (B, C', H', W')
        adv_target_khot = _to_khot(imagenet_target_per_image, num_classes=adversary.fc.out_features)

    loss_adv = torch.tensor(0.0, device=device)
    for _ in range(k_adv):
        adv_logits = adversary(adv_input)
        if gt_alg == 1:
            loss_adv = F.cross_entropy(adv_logits, adv_target_flat)
        else:
            loss_adv = F.binary_cross_entropy_with_logits(adv_logits, adv_target_khot)
        adv_optimizer.zero_grad(set_to_none=True)
        loss_adv.backward()
        adv_optimizer.step()

    # --- Encoder outer step (1 step, fresh forward through trainable encoder, frozen receiver) ---
    backbone.train()
    # Re-assert eval on receiver layers to undo what backbone.train() just flipped.
    backbone.layer3.eval()
    backbone.layer4.eval()
    backbone.fc.eval()
    # Re-enable grads only on encoder layers.
    for m in (backbone.conv1, backbone.bn1, backbone.layer1, backbone.layer2):
        for p in m.parameters():
            p.requires_grad = True

    # Disable adversary grads: encoder outer step never updates adversary params,
    # so avoid accumulating useless gradients in them during total.backward().
    for p in adversary.parameters():
        p.requires_grad = False

    pre = backbone.encode(images)
    post_channel, _, _ = backbone.channel(pre, noise_std=snr_noise_std,
                                          gpu=device.index if device.type == "cuda" else None)

    # Receiver forward (frozen weights, but no_grad isn't needed because we need grad to flow back into encoder).
    util_logits = backbone.decode(post_channel)
    loss_util = F.cross_entropy(util_logits, firearm_target)

    # Adversary forward for the privacy term (adversary weights NOT updated here).
    if gt_alg == 1:
        B, K, Cf, Hf, Wf = pre.shape
        adv_input2 = pre.reshape(B * K, Cf, Hf, Wf)
        priv_target = adv_target_flat
        if background_mask is not None:
            # Per-stacked-image mask: slot 0 follows background_mask, slots 1..K-1
            # are always backgrounds (sampled from negative pool by construction).
            per_img_mask = torch.ones(B, K, dtype=torch.bool, device=device)
            per_img_mask[:, 0] = background_mask
            per_img_mask = per_img_mask.reshape(-1)
            adv_input2 = adv_input2[per_img_mask]
            priv_target = priv_target[per_img_mask]
        adv_logits2 = adversary(adv_input2) if adv_input2.numel() > 0 else None
        if adv_logits2 is None or adv_logits2.shape[0] == 0:
            loss_priv = torch.zeros((), device=device)
        elif priv_loss_name == "ce":
            loss_priv = priv_loss_ce(adv_logits2, priv_target)
        else:
            loss_priv = priv_loss_entropy(adv_logits2)
    else:
        post_for_priv = post_channel
        khot_for_priv = adv_target_khot
        if background_mask is not None:
            post_for_priv = post_for_priv[background_mask]
            khot_for_priv = khot_for_priv[background_mask]
        adv_logits2 = adversary(post_for_priv) if post_for_priv.shape[0] > 0 else None
        if adv_logits2 is None or adv_logits2.shape[0] == 0:
            loss_priv = torch.zeros((), device=device)
        elif priv_loss_name == "ce":
            # CE form for multilabel: -BCE
            loss_priv = -F.binary_cross_entropy_with_logits(adv_logits2, khot_for_priv)
        else:
            loss_priv = priv_loss_entropy_multilabel(adv_logits2)

    total = loss_util - lam * loss_priv  # encoder MINIMIZES this; subtract privacy term gives gradient reversal
    enc_optimizer.zero_grad(set_to_none=True)
    total.backward()
    enc_optimizer.step()

    # Restore adversary params for the next stage_b_step invocation's inner loop.
    for p in adversary.parameters():
        p.requires_grad = True

    return {
        "loss_util": loss_util.detach().item(),
        "loss_priv": loss_priv.detach().item(),
        "loss_adv": loss_adv.detach().item(),
        "loss_total": total.detach().item(),
    }
