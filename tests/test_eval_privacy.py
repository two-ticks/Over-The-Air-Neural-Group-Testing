"""Unit tests for privacy.eval_privacy helpers (Phase 1 firearm-filter fix)."""
import torch

from privacy.eval_privacy import _build_background_mask


def test_background_mask_excludes_firearm_mixing_slots():
    firearm_target = torch.tensor([1, 0, 1, 0])
    mask = _build_background_mask(firearm_target, K=1)
    assert mask.shape == (4, 1)
    # Slot 0 is firearm iff firearm_target == 1; mask True means "is background".
    assert mask[:, 0].tolist() == [False, True, False, True]


def test_background_mask_preserves_extra_slots_as_background():
    """For ITIT with background_K > 0, slots 1..K-1 are always sampled
    backgrounds; the mask should treat them as background regardless of
    firearm_target."""
    firearm_target = torch.tensor([1, 0])
    mask = _build_background_mask(firearm_target, K=3)
    assert mask.shape == (2, 3)
    # Item 0 is a firearm-mixing item: only slot 0 is excluded.
    assert mask[0].tolist() == [False, True, True]
    # Item 1 is a background-mixing item: all slots are background.
    assert mask[1].tolist() == [True, True, True]


def test_background_mask_all_firearm_batch():
    firearm_target = torch.tensor([1, 1, 1])
    mask = _build_background_mask(firearm_target, K=1)
    assert not mask.any(), "All-firearm batch with K=1 should produce no background slots"


def test_background_mask_all_background_batch():
    firearm_target = torch.tensor([0, 0, 0])
    mask = _build_background_mask(firearm_target, K=1)
    assert mask.all(), "All-background batch should produce all-True mask"


def test_background_mask_dtype_and_device():
    firearm_target = torch.tensor([0, 1])
    mask = _build_background_mask(firearm_target, K=2)
    assert mask.dtype == torch.bool
    assert mask.device == firearm_target.device
