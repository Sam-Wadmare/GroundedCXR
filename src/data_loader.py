"""
Data loading for GroundedCXR (updated for TASK 1: patient-level split).

Previously this file built train/val/test DataLoaders directly from
dataset/chest_xray/{train,val,test} using torchvision.datasets.ImageFolder,
with val/ being the official folder (only 16 images) and executing all of
its logic (including printing stats and pulling a batch) as a side effect
of being imported.

This version instead:
  - Builds TRAIN and VAL datasets from the patient-grouped split CSV
    produced by src/split.py (dataset/split.csv by default), so no
    patient's images appear in both train and val.
  - Builds the TEST dataset from the official, untouched test/ folder,
    exactly as before — the test set itself is unchanged.
  - Wraps everything in functions, so importing this module has no side
    effects. The old print/sanity-check behavior is now only run when
    this file is executed directly (`python src/data_loader.py`).

app.py does NOT import this file (it only uses src/model.py,
src/inference.py, src/gradcam.py, src/chatbot.py) — this change does not
affect the running Streamlit app.

Usage (quick sanity check):
    python src/data_loader.py
    python src/data_loader.py --dataset_dir dataset/chest_xray --split_csv dataset/split.csv --batch_size 16

Usage (as a module, e.g. from a training script):
    from src.data_loader import get_dataloaders
    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(
        dataset_dir="dataset/chest_xray",
        split_csv="dataset/split.csv",
        batch_size=16,
    )
"""

import argparse
import csv
import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# Fixed class <-> index mapping used throughout the project (matches
# src/model.py / src/inference.py: 0 = NORMAL, 1 = PNEUMONIA).
CLASS_TO_IDX = {"NORMAL": 0, "PNEUMONIA": 1}

# Same preprocessing as before: resize, (train-only) horizontal flip,
# tensor, ImageNet normalization. Unchanged from the original file.
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class SplitCSVDataset(Dataset):
    """
    A dataset built from the CSV produced by src/split.py.
    Each CSV row has: filepath, class_name, patient_id, split.
    This dataset includes only the rows whose "split" column equals
    `split_name` ("train" or "val").
    """

    def __init__(self, split_csv, split_name, transform, class_to_idx=CLASS_TO_IDX):
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.samples = []

        with open(split_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split_name:
                    self.samples.append((row["filepath"], row["class_name"]))

        if not self.samples:
            raise ValueError(
                f"No rows found for split='{split_name}' in {split_csv}. "
                "Did you run src/split.py first?"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, class_name = self.samples[idx]
        image = Image.open(filepath).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.class_to_idx[class_name]
        return image, label


def get_dataloaders(dataset_dir="dataset/chest_xray", split_csv="dataset/split.csv",
                     batch_size=16, class_to_idx=CLASS_TO_IDX):
    """
    Builds and returns (train_loader, val_loader, test_loader, class_to_idx).

    train/val come from the patient-grouped split_csv (TASK 1, src/split.py).
    test comes from the official, untouched dataset_dir/test folder.
    """
    train_dataset = SplitCSVDataset(split_csv, "train", train_transform, class_to_idx)
    val_dataset = SplitCSVDataset(split_csv, "val", val_test_transform, class_to_idx)

    test_dir = os.path.join(dataset_dir, "test")
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transform)

    # ImageFolder assigns class indices alphabetically, which already
    # matches our fixed CLASS_TO_IDX ({"NORMAL": 0, "PNEUMONIA": 1}) for
    # this dataset's folder names — but we assert it instead of assuming,
    # so a renamed folder fails loudly instead of silently mislabeling.
    assert test_dataset.class_to_idx == class_to_idx, (
        f"Test folder class_to_idx {test_dataset.class_to_idx} does not "
        f"match expected {class_to_idx}. Check the test/ class folder names."
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, class_to_idx


def main():
    parser = argparse.ArgumentParser(description="Sanity-check the patient-split DataLoaders.")
    parser.add_argument("--dataset_dir", default="dataset/chest_xray")
    parser.add_argument("--split_csv", default="dataset/split.csv")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(
        dataset_dir=args.dataset_dir,
        split_csv=args.split_csv,
        batch_size=args.batch_size,
    )

    print("Class to index:", class_to_idx)
    print("Training images:", len(train_loader.dataset))
    print("Validation images:", len(val_loader.dataset))
    print("Test images:", len(test_loader.dataset))
    print("Number of training batches:", len(train_loader))

    images, labels = next(iter(train_loader))
    print("\nBatch information:")
    print("Image shape:", images.shape)
    print("Label shape:", labels.shape)
    print("Labels:", labels)


if __name__ == "__main__":
    main()