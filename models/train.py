"""
Training script for the handwriting character recognition CNN.

Reads the .npz files written by data/synthesize.py, trains a CNN with on-the-fly
augmentation, validates each epoch, and saves the best checkpoint by validation
accuracy.

Designed to run on CPU in reasonable time. With samples_per_class=400 and 5
epochs, total time is roughly 5 to 10 minutes on a modern CPU.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.augment import AugmentationPipeline
from data.dataset import HandwritingDataset, load_classes
from models.cnn import HandwritingCNN


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_aug = AugmentationPipeline(severity="light", seed=args.seed)

    train_ds = HandwritingDataset(
        Path(args.data_dir) / "train.npz", augment=train_aug)
    val_ds = HandwritingDataset(
        Path(args.data_dir) / "val.npz", augment=None)
    classes = load_classes(args.data_dir)

    print(f"train: {len(train_ds)} samples, val: {len(val_ds)} samples, "
          f"{len(classes)} classes")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers,
                              pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers,
                            pin_memory=False)

    model = HandwritingCNN(num_classes=len(classes)).to(device)
    print(f"model params: {model.count_params():,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    history = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}",
                    leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            pbar.set_postfix(loss=f"{running_loss / total:.4f}",
                             acc=f"{correct / total:.3f}")

        scheduler.step()

        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)

        elapsed = time.time() - start
        print(f"epoch {epoch}/{args.epochs}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}  "
              f"({elapsed:.1f}s elapsed)")
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "val_acc": val_acc,
                "epoch": epoch,
            }, ckpt_dir / "best.pth")
            print(f"  saved best checkpoint, val_acc={val_acc:.3f}")

    print(f"\ntraining complete in {time.time() - start:.1f}s")
    print(f"best val_acc: {best_acc:.3f}")

    import json
    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/synthetic")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train(args)
