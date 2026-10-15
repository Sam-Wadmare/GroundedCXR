import os
import random
from PIL import Image
import matplotlib.pyplot as plt

DATASET_DIR = "dataset/chest_xray/train"

classes = ["NORMAL", "PNEUMONIA"]

fig, axes = plt.subplots(2, 4, figsize=(12, 6))

for row, class_name in enumerate(classes):

    folder = os.path.join(DATASET_DIR, class_name)

    images = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    selected_images = random.sample(images, 4)

    for col, image_name in enumerate(selected_images):

        image_path = os.path.join(folder, image_name)

        image = Image.open(image_path).convert("RGB")

        axes[row, col].imshow(image)
        axes[row, col].set_title(class_name)
        axes[row, col].axis("off")

plt.tight_layout()
plt.show()