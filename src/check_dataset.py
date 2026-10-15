import os

DATASET_DIR = "dataset/chest_xray"

for split in ["train", "val", "test"]:
    print(f"\n===== {split.upper()} =====")

    for class_name in ["NORMAL", "PNEUMONIA"]:
        folder = os.path.join(DATASET_DIR, split, class_name)

        if not os.path.exists(folder):
            print(f"{class_name}: FOLDER NOT FOUND")
            continue

        images = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        print(f"{class_name}: {len(images)} images")