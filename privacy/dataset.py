"""PrivacyTaskCoalitionDataset: like main.py's TaskCoalitionDataset_SuperImposing,
but each __getitem__ also returns a (K+1,) tensor of ImageNet class indices for the
stacked images. The Stage B / Stage C adversary trains against these labels.
"""
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.utils.data
import torchvision.datasets as datasets


class PrivacyTaskCoalitionDataset(torch.utils.data.Dataset):
    """Wraps a list of `ImageFolder` datasets (one per task) and yields:

    * `images`            : (K+1, C, H, W) stacked transforms
    * `firearm_target`    : int in {0, 1} — binary firearm label
    * `imagenet_targets`  : long tensor of shape (K+1,) — per-image ImageNet class index

    The ImageNet class index is assigned as the alphabetical position of the wnid
    among all wnids present across all task folders. This makes the index stable
    across runs as long as the on-disk class folders don't change.
    """

    def __init__(self, dataset_list: List[datasets.ImageFolder], args, split: str,
                 wnid_to_imagenet_idx=None, background_only: bool = False):
        """If `background_only=True`, the firearm (positive) samples are dropped
        from the dataset entirely — the mixing slot becomes background-only.
        This is the threat-model setting for the Stage C honest-leakage adversary,
        which should never see firearm-labeled images."""
        assert split in ("train", "val")

        first = dataset_list[0]
        self.loader = first.loader
        self.transform = first.transform
        assert first.target_transform is None, "PrivacyTaskCoalitionDataset assumes target_transform is None"
        self.args = args
        self.background_only = background_only

        # Build (or accept) the global wnid -> imagenet index mapping.
        if wnid_to_imagenet_idx is not None:
            self.wnid_to_imagenet_idx = wnid_to_imagenet_idx
            self.num_imagenet_classes = len(wnid_to_imagenet_idx)
        else:
            wnid_sources = dataset_list[1:] if background_only else dataset_list
            all_wnids = sorted({wnid for ds in wnid_sources for wnid in ds.class_to_idx.keys()})
            self.wnid_to_imagenet_idx = {w: i for i, w in enumerate(all_wnids)}
            self.num_imagenet_classes = len(all_wnids)

        # Same mixing logic as TaskCoalitionDataset_SuperImposing: positives = task 0, negatives = sampled background.
        positive_data_list = dataset_list[0].samples  # firearm
        normal_data_list = []
        for ds in dataset_list[1:]:
            normal_data_list.extend(ds.samples)
        normal_data_list = np.random.permutation(normal_data_list)

        negative_data_list = normal_data_list[: len(positive_data_list)]

        positive_target = 1
        positive_data_list = [[s[0], positive_target] for s in positive_data_list]
        negative_target = 0
        negative_data_list = [[s[0], negative_target] for s in negative_data_list]

        if background_only:
            # Drop positives — adversary's data must not contain firearm samples.
            mixing_data_list = list(negative_data_list)
        else:
            mixing_data_list = positive_data_list + negative_data_list
        if split != "val":
            mixing_data_list = list(np.random.permutation(mixing_data_list))

        self.background_K = self.args.background_K

        background_K_list = []
        for _ in range(self.background_K):
            sample = list(np.random.permutation(normal_data_list)[: len(mixing_data_list)])
            background_K_list.append(sample)

        self.dataset_samples = [mixing_data_list] + background_K_list

    def __len__(self) -> int:
        return len(self.dataset_samples[0])

    def __getitem__(self, index):
        firearm_target = int(self.dataset_samples[0][index][1])

        images = []
        imagenet_targets = []
        for folder_idx in range(self.background_K + 1):
            path, _ = self.dataset_samples[folder_idx][index]
            sample = self.loader(path)
            sample = self.transform(sample)
            images.append(sample)
            wnid = Path(path).parent.name
            imagenet_targets.append(self.wnid_to_imagenet_idx[wnid])

        images_stack = torch.stack(images)
        imagenet_targets_t = torch.tensor(imagenet_targets, dtype=torch.long)
        return images_stack, firearm_target, imagenet_targets_t
