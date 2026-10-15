import torch
from PIL import Image
from torchvision import models
import torch.nn as nn

from src.inference import predict
from src.gradcam import generate_gradcam


# -----------------------------
# 1. Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# -----------------------------
# 2. Load ResNet18
# -----------------------------
model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)


# -----------------------------
# 3. Load trained weights
# -----------------------------
model.load_state_dict(
    torch.load(
        "models/best_pneumonia_resnet18.pth",
        map_location=device
    )
)

model = model.to(device)

print("Model loaded successfully")


# -----------------------------
# 4. Choose an X-ray
# -----------------------------
image_path = "dataset/chest_xray/test/NORMAL/IM-0001-0001.jpeg"

image = Image.open(image_path).convert("RGB")


# -----------------------------
# 5. Prediction
# -----------------------------
predicted_class, confidence, probabilities = predict(
    model,
    image,
    device
)

print("\nPrediction:", predicted_class)
print("Confidence:", f"{confidence * 100:.2f}%")

print("\nClass probabilities:")

for class_name, probability in probabilities.items():
    print(
        f"{class_name}: {probability * 100:.2f}%"
    )


# -----------------------------
# 6. Grad-CAM
# -----------------------------
class_index = 0 if predicted_class == "NORMAL" else 1

cam_image = generate_gradcam(
    model,
    image,
    class_index
)

cam_image.save("gradcam_test.jpg")

print("\nGrad-CAM saved as: gradcam_test.jpg")