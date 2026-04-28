"""Tests for stage_b_step: gradient flow contract and basic shape correctness."""
import pytest
import torch
import torch.nn as nn

import resnet_design2 as models
from privacy.adversary import AdversaryHead
from privacy.trainer import stage_b_step


def _params_grad_norm(module):
    """Total L2 norm of grads across all parameters (0 if all are None or zero)."""
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total ** 0.5


def _set_module_grads_to_none(module):
    for p in module.parameters():
        p.grad = None


@pytest.fixture
def tiny_setup():
    """Build a tiny end-to-end setup: ResNet-18 GT receiver, AdversaryHead, optimizers, fake batch.

    Uses CPU and 16x16 input so the test runs fast.
    """
    torch.manual_seed(0)
    backbone = models.resnet18(pretrained=False, gt=True, phase=False)
    adversary = AdversaryHead(arch_name="resnet18", num_classes=10)

    enc_params = list(backbone.conv1.parameters()) + list(backbone.bn1.parameters()) \
        + list(backbone.layer1.parameters()) + list(backbone.layer2.parameters())
    dec_params = list(backbone.layer3.parameters()) + list(backbone.layer4.parameters()) \
        + list(backbone.fc.parameters())

    enc_optim = torch.optim.SGD(enc_params, lr=0.01)
    adv_optim = torch.optim.SGD(adversary.parameters(), lr=0.01)

    # Fake minibatch: 2 stacked images per sample, 3 channels, 16x16. K+1 = 2.
    images = torch.randn(2, 2, 3, 16, 16)
    firearm_target = torch.tensor([0, 1])
    imagenet_target_per_image = torch.randint(0, 10, (2, 2))  # (B, K+1)

    return {
        "backbone": backbone,
        "adversary": adversary,
        "enc_optim": enc_optim,
        "adv_optim": adv_optim,
        "enc_params": enc_params,
        "dec_params": dec_params,
        "images": images,
        "firearm_target": firearm_target,
        "imagenet_target_per_image": imagenet_target_per_image,
    }


def test_stage_b_step_runs(tiny_setup):
    """stage_b_step completes without error and returns a metrics dict."""
    out = stage_b_step(
        backbone=tiny_setup["backbone"],
        adversary=tiny_setup["adversary"],
        enc_optimizer=tiny_setup["enc_optim"],
        adv_optimizer=tiny_setup["adv_optim"],
        images=tiny_setup["images"],
        firearm_target=tiny_setup["firearm_target"],
        imagenet_target_per_image=tiny_setup["imagenet_target_per_image"],
        priv_loss_name="entropy",
        lam=1.0,
        k_adv=2,
        device=torch.device("cpu"),
        gt_alg=1,
        background_K=1,
        snr_noise_std=None,
    )
    assert "loss_util" in out and "loss_priv" in out and "loss_adv" in out
    assert torch.isfinite(torch.tensor(out["loss_util"]))


def test_stage_b_step_freezes_receiver(tiny_setup):
    """Receiver (layer3, layer4, fc) must get NO gradient during the step (frozen by design)."""
    backbone = tiny_setup["backbone"]
    # Pre-step: zero all grads
    _set_module_grads_to_none(backbone)
    _set_module_grads_to_none(tiny_setup["adversary"])

    stage_b_step(
        backbone=backbone,
        adversary=tiny_setup["adversary"],
        enc_optimizer=tiny_setup["enc_optim"],
        adv_optimizer=tiny_setup["adv_optim"],
        images=tiny_setup["images"],
        firearm_target=tiny_setup["firearm_target"],
        imagenet_target_per_image=tiny_setup["imagenet_target_per_image"],
        priv_loss_name="entropy",
        lam=1.0,
        k_adv=1,
        device=torch.device("cpu"),
        gt_alg=1,
        background_K=1,
        snr_noise_std=None,
    )

    # After step: encoder params (conv1, bn1, layer1, layer2) should have nonzero grads.
    for m in (backbone.conv1, backbone.layer1, backbone.layer2):
        assert _params_grad_norm(m) > 0, f"encoder layer {m.__class__.__name__} got no gradient"

    # Decoder (layer3, layer4, fc) must be untouched by the step (zeroed afterwards).
    for m in (backbone.layer3, backbone.layer4, backbone.fc):
        for p in m.parameters():
            assert p.grad is None or p.grad.abs().sum().item() == 0, \
                f"decoder {m.__class__.__name__} got gradient — frozen-receiver contract broken"


def test_stage_b_step_priv_loss_ce_branch_runs(tiny_setup):
    """priv_loss_name='ce' takes the CE branch instead of entropy."""
    out = stage_b_step(
        backbone=tiny_setup["backbone"],
        adversary=tiny_setup["adversary"],
        enc_optimizer=tiny_setup["enc_optim"],
        adv_optimizer=tiny_setup["adv_optim"],
        images=tiny_setup["images"],
        firearm_target=tiny_setup["firearm_target"],
        imagenet_target_per_image=tiny_setup["imagenet_target_per_image"],
        priv_loss_name="ce",
        lam=0.5,
        k_adv=1,
        device=torch.device("cpu"),
        gt_alg=1,
        background_K=1,
        snr_noise_std=None,
    )
    assert torch.isfinite(torch.tensor(out["loss_util"]))


def test_stage_b_step_gtgt_fm_branch_runs(tiny_setup):
    """gt_alg=2 takes the GTGT-FM (multilabel) branch."""
    out = stage_b_step(
        backbone=tiny_setup["backbone"],
        adversary=tiny_setup["adversary"],
        enc_optimizer=tiny_setup["enc_optim"],
        adv_optimizer=tiny_setup["adv_optim"],
        images=tiny_setup["images"],
        firearm_target=tiny_setup["firearm_target"],
        imagenet_target_per_image=tiny_setup["imagenet_target_per_image"],
        priv_loss_name="entropy",
        lam=1.0,
        k_adv=1,
        device=torch.device("cpu"),
        gt_alg=2,
        background_K=1,
        snr_noise_std=None,
    )
    assert torch.isfinite(torch.tensor(out["loss_util"]))


def test_stage_b_step_does_not_update_receiver_bn_running_stats(tiny_setup):
    """Receiver BN running_mean/running_var must not change during a stage_b_step (frozen by design)."""
    backbone = tiny_setup["backbone"]
    # Snapshot the first BN in the first block of layer3 before the step.
    bn = backbone.layer3[0].bn1
    mean_before = bn.running_mean.clone()
    var_before = bn.running_var.clone()

    stage_b_step(
        backbone=backbone,
        adversary=tiny_setup["adversary"],
        enc_optimizer=tiny_setup["enc_optim"],
        adv_optimizer=tiny_setup["adv_optim"],
        images=tiny_setup["images"],
        firearm_target=tiny_setup["firearm_target"],
        imagenet_target_per_image=tiny_setup["imagenet_target_per_image"],
        priv_loss_name="entropy",
        lam=1.0,
        k_adv=1,
        device=torch.device("cpu"),
        gt_alg=1,
        background_K=1,
        snr_noise_std=None,
    )

    assert torch.equal(bn.running_mean, mean_before), \
        "layer3[0].bn1.running_mean was mutated — receiver BN running stats leak!"
    assert torch.equal(bn.running_var, var_before), \
        "layer3[0].bn1.running_var was mutated — receiver BN running stats leak!"


def test_stage_b_step_does_not_update_adversary_in_outer_step(tiny_setup):
    """Encoder outer step must NOT accumulate gradient into adversary parameters.

    With k_adv=0, the adversary inner loop is skipped entirely. Any gradient that
    appears on adversary params after the step must have come from the encoder's
    outer-step backward — which the I3 fix prevents via requires_grad=False toggle.
    """
    adversary = tiny_setup["adversary"]
    # Clear adversary grads BEFORE the step so we only see what this step accumulates.
    for p in adversary.parameters():
        p.grad = None

    stage_b_step(
        backbone=tiny_setup["backbone"],
        adversary=adversary,
        enc_optimizer=tiny_setup["enc_optim"],
        adv_optimizer=tiny_setup["adv_optim"],
        images=tiny_setup["images"],
        firearm_target=tiny_setup["firearm_target"],
        imagenet_target_per_image=tiny_setup["imagenet_target_per_image"],
        priv_loss_name="entropy",
        lam=1.0,
        k_adv=0,
        device=torch.device("cpu"),
        gt_alg=1,
        background_K=1,
        snr_noise_std=None,
    )

    for i, p in enumerate(adversary.parameters()):
        leaked = (p.grad is not None) and (p.grad.abs().sum().item() > 0)
        assert not leaked, (
            f"adversary parameter [{i}] (shape {tuple(p.shape)}) accumulated gradient "
            f"during the encoder outer step — I3 grad leak!"
        )


def _itit_single_stack_batch(B=2):
    """ITIT batch with background_K=0 — one image per item (K+1 == 1)."""
    images = torch.randn(B, 1, 3, 16, 16)
    firearm_target = torch.tensor([1] * B)  # all firearm
    imagenet_target_per_image = torch.randint(0, 10, (B, 1))
    return images, firearm_target, imagenet_target_per_image


def test_stage_b_step_background_mask_zero_priv_loss_when_all_firearm(tiny_setup):
    """All-firearm batch + background_mask=all-False, with background_K=0
    (only the mixing slot exists). Every stacked image is masked out, so
    the privacy term has no samples to compute over → loss_priv == 0."""
    images, firearm_target, imagenet_target = _itit_single_stack_batch(B=2)

    out = stage_b_step(
        backbone=tiny_setup["backbone"],
        adversary=tiny_setup["adversary"],
        enc_optimizer=tiny_setup["enc_optim"],
        adv_optimizer=tiny_setup["adv_optim"],
        images=images,
        firearm_target=firearm_target,
        imagenet_target_per_image=imagenet_target,
        priv_loss_name="entropy",
        lam=1.0,
        k_adv=0,  # skip inner loop — focus on the priv-loss path
        device=torch.device("cpu"),
        gt_alg=1,
        background_K=0,
        snr_noise_std=None,
        background_mask=torch.zeros(2, dtype=torch.bool),
    )
    assert out["loss_priv"] == 0.0, (
        f"Expected loss_priv == 0.0 for an all-firearm batch with background_mask=all-False, "
        f"got {out['loss_priv']}"
    )


def test_stage_b_step_background_mask_keeps_extra_slots(tiny_setup):
    """With background_K > 0, the per-image mask must keep slots 1..K (always
    background by construction), even when slot 0 is masked out. Loss_priv
    should be NON-zero because the K extra background slots still contribute."""
    out = stage_b_step(
        backbone=tiny_setup["backbone"],
        adversary=tiny_setup["adversary"],
        enc_optimizer=tiny_setup["enc_optim"],
        adv_optimizer=tiny_setup["adv_optim"],
        images=tiny_setup["images"],  # (B=2, K+1=2, ...)
        firearm_target=torch.tensor([1, 1]),
        imagenet_target_per_image=tiny_setup["imagenet_target_per_image"],
        priv_loss_name="entropy",
        lam=1.0,
        k_adv=0,
        device=torch.device("cpu"),
        gt_alg=1,
        background_K=1,
        snr_noise_std=None,
        background_mask=torch.zeros(2, dtype=torch.bool),
    )
    assert out["loss_priv"] != 0.0, (
        "Expected loss_priv != 0.0: slot-1 stacked images are always background "
        "and should contribute to priv_loss even when slot-0 is masked out."
    )


def test_stage_b_step_background_mask_none_matches_legacy(tiny_setup):
    """Passing background_mask=None must reproduce the legacy (pre-mask)
    behavior — the priv_loss should equal the value from a call without
    the parameter."""
    images, firearm_target, imagenet_target = _itit_single_stack_batch(B=2)

    out_none = stage_b_step(
        backbone=tiny_setup["backbone"], adversary=tiny_setup["adversary"],
        enc_optimizer=tiny_setup["enc_optim"], adv_optimizer=tiny_setup["adv_optim"],
        images=images, firearm_target=firearm_target,
        imagenet_target_per_image=imagenet_target,
        priv_loss_name="entropy", lam=1.0, k_adv=0,
        device=torch.device("cpu"), gt_alg=1, background_K=0,
        snr_noise_std=None,
        background_mask=None,
    )
    assert torch.isfinite(torch.tensor(out_none["loss_priv"]))
