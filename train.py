"""
Training loop for CBIS-DDSM-style classification.

Run:
    python train.py --data_root /path/to/CBIS_DDSM_ROOT
"""

import argparse
import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config import (
    DATA_ROOT,
    LEARNING_RATE,
    NUM_EPOCHS,
    OUTPUT_DIR,
    CHECKPOINT_PATH,
    RANDOM_SEED,
    WEIGHT_DECAY,
)
from dataset import create_dataloaders
from model import build_model
from utils import compute_classification_metrics, save_checkpoint, load_checkpoint


def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model, loader, criterion, optimizer, device
) -> Tuple[float, float]:
    model.train()
    epoch_loss = 0.0

    all_targets = []
    all_preds = []

    for images, targets in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        all_targets.extend(targets.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

    epoch_loss /= len(loader.dataset)
    metrics = compute_classification_metrics(all_targets, all_preds, average="binary")
    return epoch_loss, metrics["accuracy"]


def evaluate(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0.0

    all_targets = []
    all_preds = []
    all_scores = []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Eval", leave=False):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            epoch_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)

            all_targets.extend(targets.detach().cpu().tolist())
            all_preds.extend(preds.detach().cpu().tolist())
            all_scores.extend(probs.detach().cpu().tolist())

    epoch_loss /= len(loader.dataset)
    metrics = compute_classification_metrics(
        all_targets, all_preds, all_scores, average="binary"
    )
    return epoch_loss, metrics


def main(args):
    set_seed(RANDOM_SEED)

    data_root = args.data_root or DATA_ROOT
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Using data_root = {data_root}")

    train_loader, val_loader, test_loader = create_dataloaders(data_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_val_f1 = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        print(f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        print(
            "Val   loss: {:.4f}, acc: {:.4f}, prec: {:.4f}, recall: {:.4f}, "
            "f1: {:.4f}, roc_auc: {:.4f}".format(
                val_loss,
                val_metrics.get("accuracy", 0.0),
                val_metrics.get("precision", 0.0),
                val_metrics.get("recall", 0.0),
                val_metrics.get("f1", 0.0),
                val_metrics.get("roc_auc", 0.0),
            )
        )

        if val_metrics.get("f1", 0.0) > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            save_checkpoint(model, CHECKPOINT_PATH)
            print(f"Saved new best model to {CHECKPOINT_PATH}")

    print("\nEvaluating best model on test set...")
    best_model = build_model(pretrained=False).to(device)
    best_model = load_checkpoint(best_model, CHECKPOINT_PATH, map_location=device)

    test_loss, test_metrics = evaluate(best_model, test_loader, criterion, device)
    print(
        "Test  loss: {:.4f}, acc: {:.4f}, prec: {:.4f}, recall: {:.4f}, "
        "f1: {:.4f}, roc_auc: {:.4f}".format(
            test_loss,
            test_metrics.get("accuracy", 0.0),
            test_metrics.get("precision", 0.0),
            test_metrics.get("recall", 0.0),
            test_metrics.get("f1", 0.0),
            test_metrics.get("roc_auc", 0.0),
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Root directory of images (overrides config.DATA_ROOT)",
    )
    args = parser.parse_args()
    main(args)