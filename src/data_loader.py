import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# -----------------------------
# 1. Dataset paths
# -----------------------------

DATASET_DIR = "dataset/chest_xray"

train_dir = f"{DATASET_DIR}/train"
val_dir = f"{DATASET_DIR}/val"
test_dir = f"{DATASET_DIR}/test"


# -----------------------------
# 2. Image transformations
# -----------------------------

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# 3. Load datasets
# -----------------------------

train_dataset = datasets.ImageFolder(
    train_dir,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    val_dir,
    transform=val_test_transform
)

test_dataset = datasets.ImageFolder(
    test_dir,
    transform=val_test_transform
)


# -----------------------------
# 4. Create DataLoaders
# -----------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)


# -----------------------------
# 5. Print information
# -----------------------------

print("Classes:", train_dataset.classes)

print("Class to index:", train_dataset.class_to_idx)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Test images:", len(test_dataset))

print("Number of training batches:", len(train_loader))


# -----------------------------
# 6. Check one batch
# -----------------------------

images, labels = next(iter(train_loader))

print("\nBatch information:")
print("Image shape:", images.shape)
print("Label shape:", labels.shape)
print("Labels:", labels)