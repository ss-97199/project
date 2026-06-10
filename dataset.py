"""
Dataset and dataloader utilities for CBIS-DDSM-like directory structure.
"""

import os
from typing import Tuple, List

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T

from config import IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS


class MammogramDataset(Dataset):
    """
    Folder-based dataset.

    Expected structure:
        root/
            benign/
                *.png, *.jpg, ...
            malignant/
                *.png, *.jpg, ...
    """

    def __init__(self, root: str, train: bool = True, augment: bool = True):
        self.root = root
        self.image_paths: List[str] = []
        self.labels: List[int] = []

        class_names = sorted(
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
        )
        if len(class_names) != 2:
            raise ValueError(
                f"Expected 2 class folders under {root}, found {class_names}"
            )

        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        print(f"Classes: {self.class_to_idx}")

        for cls in class_names:
            cls_dir = os.path.join(root, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
                ):
                    self.image_paths.append(os.path.join(cls_dir, fname))
                    self.labels.append(self.class_to_idx[cls])

        # Standard ImageNet normalization, even though mammo is grayscale.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_transforms = T.Compose(
            [
                T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
                T.RandomAffine(
                    degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)
                ),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

        eval_transforms = T.Compose(
            [
                T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

        self.transform = train_transforms if (train and augment) else eval_transforms

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        return img, label


def create_dataloaders(
    data_root: str,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders from a single root directory
    by random 70/15/15 split.
    """
    full_dataset = MammogramDataset(data_root, train=True, augment=True)

    n_total = len(full_dataset)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test]
    )

    # For val/test, use eval transforms (no augmentation)
    eval_dataset = MammogramDataset(data_root, train=False, augment=False)
    val_ds.dataset.transform = eval_dataset.transform
    test_ds.dataset.transform = eval_dataset.transform

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader