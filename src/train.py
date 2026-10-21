"""
TASK 3: Generic training script for GroundedCXR.

Trains either:
  - "resnet18": the project's existing architecture (src/model.py's
    create_model() — ImageNet-pretrained, only layer4 + fc fine-tuned),
    retrained here on the NEW patient-grouped split (src/split.py),
    unlike the original best_pneumonia_resnet18.pth checkpoint which
    predates that split.
  - "simple_cnn": a from-scratch baseline (src/simple_cnn.py's
    SimpleCNN() — no pretrained weights at all).

Both use the SAME data (src/data_loader.py: patient-grouped train/val
from split.csv), the SAME augmentation (train_transform inside
data_loader.py), and the SAME class-weighted cross-entropy loss, so the
two can be compared fairly afterward in results/comparison.md.

This script intentionally does NOT touch the test set at all — only
train/val. Run src/evaluate.py afterward, pointing --checkpoint at
whichever file this script saves, to get test-set metrics. Keeping
training and test-evaluation in separate scripts means training can
never accidentally peek at test data.

Usage:
    python -m src.train --model simple_cnn --epochs 10
    python -m src.train --model resnet18 --epochs 5 --output models/best_resnet18_new_split.pth
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from src.data_loader import get_dataloaders
from src.model import create_model
from src.simple_cnn import SimpleCNN


def set_seed(seed):
    """Sets every RNG this script touches, for reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(model_name):
    if model_name == "resnet18":
        return create_model()
    elif model_name == "simple_cnn":
        return SimpleCNN(num_classes=2)
    raise ValueError(f"Unknown --model: {model_name}")


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compute_class_weights(train_dataset, class_to_idx, num_classes=2):
    """
    "Balanced" class weights: weight_i = total_samples / (num_classes * count_i).

    This gives each class equal total influence on the loss regardless
    of how many examples it has, countering the roughly 2.9:1
    PNEUMONIA:NORMAL imbalance in the training split. Matches the
    README's documented "class-weighted cross-entropy" approach for the
    original ResNet18 training, applied here identically to both models
    for a fair comparison.
    """
    counts = [0] * num_classes
    for _filepath, class_name in train_dataset.samples:
        counts[class_to_idx[class_name]] += 1

    total = sum(counts)
    weights = [total / (num_classes * c) if c > 0 else 0.0 for c in counts]
    return torch.tensor(weights, dtype=torch.float32), counts


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(
        description="TASK 3: Train ResNet18 or a from-scratch baseline CNN on the patient-grouped split."
    )
    parser.add_argument("--model", choices=["resnet18", "simple_cnn"], required=True)
    parser.add_argument("--dataset_dir", default="dataset/chest_xray")
    parser.add_argument("--split_csv", default="dataset/split.csv")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None,
                         help="Where to save the best checkpoint (default: models/best_<model>.pth)")
    parser.add_argument("--history_json", default=None,
                         help="Where to save per-epoch training history (default: results/train_history_<model>.json)")
    args = parser.parse_args()

    set_seed(args.seed)

    output_path = args.output or f"models/best_{args.model}.pth"
    history_path = args.history_json or f"results/train_history_{args.model}.json"
    for path in (output_path, history_path):
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    train_loader, val_loader, _test_loader, class_to_idx = get_dataloaders(
        dataset_dir=args.dataset_dir,
        split_csv=args.split_csv,
        batch_size=args.batch_size,
    )
    print(f"Train images: {len(train_loader.dataset)}, Val images: {len(val_loader.dataset)}")

    class_weights, class_counts = compute_class_weights(train_loader.dataset, class_to_idx)
    print(f"Train class counts {class_to_idx}: {class_counts}")
    print(f"Class weights (balanced): {class_weights.tolist()}")

    print(f"Building model: {args.model}")
    model = build_model(args.model).to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}  Trainable parameters: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    best_val_accuracy = -1.0
    history = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate_one_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch}/{args.epochs} - "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        })

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), output_path)
            print(f"  -> New best val_accuracy={val_acc:.4f}, saved to {output_path}")

    training_time_seconds = time.time() - start_time

    summary = {
        "model": args.model,
        "device": str(device),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "train_class_counts": {k: class_counts[v] for k, v in class_to_idx.items()},
        "class_weights_balanced": class_weights.tolist(),
        "best_val_accuracy": best_val_accuracy,
        "training_time_seconds": training_time_seconds,
        "checkpoint_path": output_path,
        "history": history,
    }

    with open(history_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved training history to {history_path}")
    print(f"Best checkpoint saved to {output_path} (val_accuracy={best_val_accuracy:.4f})")
    print(f"Total training time: {training_time_seconds:.1f} seconds")


if __name__ == "__main__":
    main()
