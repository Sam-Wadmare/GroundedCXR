import numpy as np
import torch

from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# Same image size used during training
IMAGE_SIZE = (224, 224)


def generate_gradcam(model, image: Image.Image, predicted_class_index: int):
    """
    Generate a Grad-CAM visualization for the model prediction.
    """

    model.eval()

    # Convert image to RGB and resize
    image = image.convert("RGB")
    resized_image = image.resize(IMAGE_SIZE)

    # Convert image to numpy format for visualization
    rgb_image = np.array(resized_image).astype(np.float32) / 255.0

    # Preprocess image exactly like training
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    input_tensor = transform(resized_image).unsqueeze(0)

    # ResNet18 final convolutional block
    target_layers = [model.layer4[-1]]

    # Grad-CAM
    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    targets = [
        ClassifierOutputTarget(predicted_class_index)
    ]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    # Overlay heatmap on X-ray
    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    return Image.fromarray(visualization)