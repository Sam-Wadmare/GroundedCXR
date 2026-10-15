import os

import torch
import torch.nn as nn
from torchvision import models

# Added: used by inference.py to map class index -> label
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def create_model():
    # Load ResNet18 with ImageNet pretrained weights
    model = models.resnet18(weights="DEFAULT")

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the final ResNet block
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace the original ImageNet classifier
    # Original: 1000 classes
    # Ours: 2 classes
    num_features = model.fc.in_features

    model.fc = nn.Linear(num_features, 2)

    return model


# Added: builds the architecture via create_model() and loads your
# trained checkpoint into it. This is what app.py actually calls.
def load_trained_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at '{checkpoint_path}'. "
            "Please place 'best_pneumonia_resnet18.pth' inside the 'models/' folder."
        )

    model = create_model()

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Support both a raw state_dict and a dict-wrapped checkpoint
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Strip a possible "module." prefix left over from DataParallel training
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[len("module."):] if k.startswith("module.") else k
        cleaned_state_dict[new_key] = v

    model.load_state_dict(cleaned_state_dict)
    model.to(device)
    model.eval()
    return model