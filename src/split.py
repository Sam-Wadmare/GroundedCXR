"""
TASK 1: Patient-level train/validation split for GroundedCXR.

Why this exists:
The official Kaggle "chest_xray" dataset's train/ folder contains
multiple images per patient (confirmed by src/inspect_filenames.py on
this project's actual data: 244/1000 NORMAL patients and 1005/1635
PNEUMONIA patients have more than one image). A plain random split of
train/ (e.g. torchvision.datasets.ImageFolder + a random train/val
split) can put the SAME patient's images in both train and validation,
which inflates validation accuracy without the model actually
generalizing better. This is a documented issue with this dataset and
is a likely explanation for the ~99% validation accuracy vs 81.57% test
accuracy gap mentioned in the project README.

What this script does:
  - Reads ONLY the official train/ folder.
  - Extracts a patient/group ID per image (confirmed reliable by
    src/inspect_filenames.py: 100% match rate on this dataset):
      PNEUMONIA: "personXXX_bacteria_YYY.jpeg" / "personXXX_virus_YYY.jpeg"
        -> group ID = "pneumonia_XXX"
      NORMAL: "IM-XXXX-YYYY.jpeg" or "NORMAL2-IM-XXXX-YYYY.jpeg"
        -> group ID = "normal_XXXX"
  - Uses sklearn's GroupShuffleSplit to split train/ into a new
    TRAIN and VAL set such that no group (patient) appears in both.
  - Saves the result to a CSV with columns: filepath, class_name,
    patient_id, split.
  - Prints class counts per split and a leakage check (must be 0).
  - Does NOT touch the official test/ folder at all. Test set is
    unchanged.

If a filename in either class doesn't match the expected pattern, this
script STOPS with an error rather than silently guessing a group for
it, since an incorrect patient ID would defeat the whole point of this
split.

Usage:
    python src/split.py
    python src/split.py --dataset_dir dataset/chest_xray --val_size 0.15 --seed 42 --output_csv dataset/split.csv
"""

import argparse
import csv
import os
import re

from sklearn.model_selection import GroupShuffleSplit

PNEUMONIA_PATTERN = re.compile(r"^person(\d+)_(bacteria|virus)_(\d+)", re.IGNORECASE)
NORMAL_PATTERN = re.compile(r"^(?:NORMAL2-)?IM-(\d+)-(\d+)", re.IGNORECASE)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def list_images(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    )


def extract_patient_id(filename, class_name):
    """
    Returns a string group/patient ID for a filename, or None if the
    filename does not match the expected pattern for its class.
    """
    if class_name == "PNEUMONIA":
        m = PNEUMONIA_PATTERN.match(filename)
        if m:
            return f"pneumonia_{m.group(1)}"
    elif class_name == "NORMAL":
        m = NORMAL_PATTERN.match(filename)
        if m:
            return f"normal_{m.group(1)}"
    return None


def build_records(dataset_dir):
    """
    Walks dataset_dir/train/{NORMAL,PNEUMONIA} and returns a list of
    dicts: {filepath, class_name, patient_id}.

    Raises ValueError (refuses to guess) if any filename doesn't match
    the expected pattern for its class.
    """
    records = []
    train_dir = os.path.join(dataset_dir, "train")

    for class_name in ["NORMAL", "PNEUMONIA"]:
        folder = os.path.join(train_dir, class_name)
        filenames = list_images(folder)

        if not filenames:
            raise ValueError(f"No images found in {folder}. Check --dataset_dir.")

        for fname in filenames:
            patient_id = extract_patient_id(fname, class_name)
            if patient_id is None:
                raise ValueError(
                    f"Could not extract a patient ID from '{fname}' in class "
                    f"'{class_name}'. Refusing to guess a grouping for it — "
                    "inspect this filename manually (re-run "
                    "src/inspect_filenames.py) before re-running split.py."
                )
            records.append({
                "filepath": os.path.join(folder, fname),
                "class_name": class_name,
                "patient_id": patient_id,
            })

    return records


def leakage_check(train_records, val_records):
    """Returns (overlap_count, overlap_ids) of patient IDs in BOTH splits."""
    train_ids = {r["patient_id"] for r in train_records}
    val_ids = {r["patient_id"] for r in val_records}
    overlap = train_ids & val_ids
    return len(overlap), overlap


def print_class_counts(name, records):
    counts = {"NORMAL": 0, "PNEUMONIA": 0}
    for r in records:
        counts[r["class_name"]] += 1
    print(f"{name}: NORMAL={counts['NORMAL']}, PNEUMONIA={counts['PNEUMONIA']}, "
          f"TOTAL={len(records)}")


def main():
    parser = argparse.ArgumentParser(description="Create a patient-level train/val split (TASK 1).")
    parser.add_argument("--dataset_dir", default="dataset/chest_xray",
                         help="Path to the dataset root (default: dataset/chest_xray)")
    parser.add_argument("--val_size", type=float, default=0.15,
                         help="Approximate fraction of train-folder IMAGES to hold "
                              "out as validation, respecting patient grouping (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_csv", default="dataset/split.csv",
                         help="Where to save the resulting split (default: dataset/split.csv)")
    args = parser.parse_args()

    print(f"Reading official train folder: {args.dataset_dir}/train ...")
    records = build_records(args.dataset_dir)
    print(f"Total images found: {len(records)}")
    print(f"Unique patient/group IDs: {len({r['patient_id'] for r in records})}")

    groups = [r["patient_id"] for r in records]
    indices = list(range(len(records)))

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(indices, groups=groups))

    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]

    for r in train_records:
        r["split"] = "train"
    for r in val_records:
        r["split"] = "val"

    print("\n----- CLASS COUNTS -----")
    print_class_counts("TRAIN", train_records)
    print_class_counts("VAL", val_records)

    overlap_count, overlap_ids = leakage_check(train_records, val_records)
    print("\n----- LEAKAGE CHECK -----")
    print(f"Patients present in BOTH train and val: {overlap_count}")
    if overlap_count > 0:
        print("WARNING: leakage detected! This should never happen with "
              "GroupShuffleSplit — stopping without writing the CSV.")
        raise RuntimeError(f"Leakage check failed: {overlap_count} overlapping patient IDs.")
    print("OK: no patient overlap between train and val.")

    all_records = train_records + val_records
    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "class_name", "patient_id", "split"])
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\nSaved split to {args.output_csv}")
    print("NOTE: the official test/ folder was not read or modified. "
          "It remains the test set, unchanged.")


if __name__ == "__main__":
    main()
