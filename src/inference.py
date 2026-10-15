"""
Inference utilities for GroundedCXR.

Handles preprocessing (identical to training) and running the trained
ResNet18 model on an uploaded chest X-ray image.
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.model import CLASS_NAMES

IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transform() -> transforms.Compose:
    """
    Returns the exact preprocessing pipeline used at training time:
    resize to 224x224 -> force 3-channel -> tensor -> ImageNet normalization.
    """
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Applies training-time preprocessing to a PIL image.

    Returns:
        torch.Tensor of shape (1, 3, 224, 224)
    """
    image = image.convert("RGB")
    transform = get_transform()
    tensor = transform(image)
    return tensor.unsqueeze(0)


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
) -> Tuple[str, float, torch.Tensor]:
    """
    Runs inference on a preprocessed tensor.

    Returns:
        predicted_label: "NORMAL" or "PNEUMONIA"
        confidence: softmax probability of the predicted class, as a
            percentage (0-100)
        probabilities: full softmax probability vector (CPU tensor)
    """
    input_tensor = input_tensor.to(device)
    logits = model(input_tensor)
    probabilities = F.softmax(logits, dim=1).squeeze(0)

    predicted_idx = int(torch.argmax(probabilities).item())
    predicted_label = CLASS_NAMES[predicted_idx]
    confidence = float(probabilities[predicted_idx].item()) * 100.0

    return predicted_label, confidence, probabilities.cpu()


def build_prediction_context(predicted_label: str, probabilities: torch.Tensor) -> dict:
    """
    Packages the model's raw prediction into a clean, structured dictionary
    that is independent of the Streamlit UI. This is what a future chatbot
    (or any other consumer) would read instead of re-deriving it from the
    display logic.

    Args:
        predicted_label: "NORMAL" or "PNEUMONIA", as returned by predict().
        probabilities: the full softmax probability vector (CPU tensor,
            values in [0, 1]), as returned by predict().

    Returns:
        dict with keys:
            prediction           (str)   "NORMAL" or "PNEUMONIA"
            confidence            (float) probability of the predicted class, 0-1
            normal_probability    (float) probability of NORMAL, 0-1
            pneumonia_probability (float) probability of PNEUMONIA, 0-1
    """
    normal_prob = float(probabilities[CLASS_NAMES.index("NORMAL")].item())
    pneumonia_prob = float(probabilities[CLASS_NAMES.index("PNEUMONIA")].item())

    return {
        "prediction": predicted_label,
        "confidence": round(max(normal_prob, pneumonia_prob), 4),
        "normal_probability": round(normal_prob, 4),
        "pneumonia_probability": round(pneumonia_prob, 4),
    }