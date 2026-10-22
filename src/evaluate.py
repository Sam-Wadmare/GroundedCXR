"""
TASK 2: Evaluation script for GroundedCXR's ResNet18 classifier.

Loads a trained checkpoint, runs it on the VALIDATION set (from the
patient-grouped split produced by src/split.py) to choose a decision
threshold, then applies that exact threshold ONCE to the official TEST
set (untouched, from dataset/chest_xray/test/). Reports accuracy,
per-class precision/recall/F1, confusion matrix, and ROC-AUC at both the
default 0.5 threshold and the tuned threshold, and saves everything to
results/.

Threshold selection rule (chosen on validation only, never on test):
    Among thresholds where sensitivity (pneumonia recall) >= min_sensitivity
    (default 0.95), pick the one with the best specificity (normal recall).

    Why this rule: the project's known issue is high pneumonia recall
    (98.97% at 0.5) but poor normal recall/specificity (52.56%) — too
    many false positives on normal X-rays. We don't want to trade away
    sensitivity (missing a pneumonia case is the worse error), so we
    fix a sensitivity floor and search for the threshold that reduces
    false alarms the most within that constraint.

Usage (run from the project root so `from src.X import Y` resolves):
    python src/evaluate.py
    python src/evaluate.py \
        --dataset_dir dataset/chest_xray --split_csv dataset/split.csv \
        --checkpoint models/best_pneumonia_resnet18.pth --batch_size 16 \
        --seed 42 --results_dir results --min_sensitivity 0.95 \
        --uncertain_band 0.4 0.6
"""

import argparse
import json
import os
import random

import matplotlib
matplotlib.use("Agg")  # headless-safe backend: works on Colab / CI / no display
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from src.data_loader import get_dataloaders
from src.model import CLASS_NAMES
from src.train import build_model


def set_seed(seed):
    """Sets every RNG this script touches, for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def get_probs_and_labels(model, loader, device):
    """
    Runs the model over every batch in `loader` and collects:
        y_true: np.array of true labels (0=NORMAL, 1=PNEUMONIA)
        y_prob: np.array of the model's predicted PNEUMONIA probability
                (softmax output at index 1), one per image
    """
    model.eval()
    all_labels = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1]  # P(PNEUMONIA)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    return y_true, y_prob


def compute_metrics(y_true, y_prob, threshold):
    """
    Computes accuracy, per-class precision/recall/F1, confusion matrix,
    and ROC-AUC for one decision threshold. ROC-AUC itself does not
    depend on the threshold (it's computed from the raw probabilities),
    it's included here just so every metrics dict is self-contained.
    """
    y_pred = (y_prob >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    auc = roc_auc_score(y_true, y_prob)

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # pneumonia recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # normal recall

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist(),  # rows/cols ordered [NORMAL, PNEUMONIA]
        "normal": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0]),
        },
        "pneumonia": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1]),
        },
        "sensitivity_pneumonia_recall": float(sensitivity),
        "specificity_normal_recall": float(specificity),
    }


def tune_threshold(y_true_val, y_prob_val, min_sensitivity=0.95):
    """
    Chosen ONLY on the validation set — never looks at test data.

    Scans every unique probability value seen on validation (plus 0.0
    and 1.0 as boundaries) as a candidate threshold, keeps only the
    candidates whose sensitivity (pneumonia recall) is >= min_sensitivity,
    and returns the one with the best specificity among those.

    If no threshold on validation reaches min_sensitivity, falls back to
    0.5 and says so explicitly (this should be rare in practice, since
    0.5 already gives ~99% sensitivity on this model).
    """
    candidates = np.unique(np.concatenate([y_prob_val, [0.0, 1.0]]))

    best_threshold = None
    best_specificity = -1.0
    reached_min_sensitivity = False

    for t in candidates:
        m = compute_metrics(y_true_val, y_prob_val, t)
        if m["sensitivity_pneumonia_recall"] >= min_sensitivity:
            reached_min_sensitivity = True
            if m["specificity_normal_recall"] > best_specificity:
                best_specificity = m["specificity_normal_recall"]
                best_threshold = t

    if not reached_min_sensitivity:
        print(f"WARNING: no threshold on the validation set reached "
              f"sensitivity >= {min_sensitivity}. Falling back to 0.5.")
        return 0.5, False

    return float(best_threshold), True


def plot_confusion_matrix(cm, title, out_path, class_names=CLASS_NAMES):
    """Saves a simple confusion matrix heatmap."""
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_roc_curve(y_true, y_prob, auc, title, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def uncertain_band_table(y_true, y_prob, lower, upper):
    """
    Optional "uncertain, refer" band: predictions with probability inside
    [lower, upper] are treated as too close to call and set aside.
    Coverage = fraction of cases NOT in that band (i.e. confidently
    classified). Accuracy is computed only on the confident cases, at
    the default 0.5 threshold, to show the accuracy/coverage trade-off.
    """
    uncertain_mask = (y_prob >= lower) & (y_prob <= upper)
    confident_mask = ~uncertain_mask

    coverage = float(confident_mask.mean())

    if confident_mask.sum() == 0:
        confident_accuracy = None
    else:
        y_pred_confident = (y_prob[confident_mask] >= 0.5).astype(int)
        confident_accuracy = float(accuracy_score(y_true[confident_mask], y_pred_confident))

    return {
        "band": [float(lower), float(upper)],
        "coverage": coverage,
        "accuracy_on_confident_cases": confident_accuracy,
        "n_uncertain": int(uncertain_mask.sum()),
        "n_total": int(len(y_true)),
    }


def load_checkpoint_for_model(model_name, checkpoint_path, device):
    """
    Builds the architecture named by `model_name` ("resnet18" or
    "simple_cnn", via src.train.build_model) and loads `checkpoint_path`'s
    weights into it.

    This generalizes src.model.load_trained_model() (which always builds
    a ResNet18, regardless of what checkpoint you give it) so evaluate.py
    can check checkpoints produced by EITHER model src/train.py can train
    — trying to load a SimpleCNN checkpoint into a ResNet18 architecture
    (or vice versa) fails loudly with a key mismatch, since the two
    architectures don't share parameter names.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. Check --checkpoint "
            "and make sure --model matches the architecture it was trained with."
        )

    model = build_model(model_name)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Support both a raw state_dict and a dict-wrapped checkpoint, same as
    # src.model.load_trained_model does for ResNet18 checkpoints.
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Strip a possible "module." prefix left over from DataParallel training.
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[len("module."):] if k.startswith("module.") else k
        cleaned_state_dict[new_key] = v

    model.load_state_dict(cleaned_state_dict)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="TASK 2: Evaluate GroundedCXR ResNet18 (metrics + validation-tuned threshold)."
    )
    parser.add_argument("--model", choices=["resnet18", "simple_cnn"], default="resnet18",
                         help="Which architecture --checkpoint was trained with (default: resnet18)")
    parser.add_argument("--dataset_dir", default="dataset/chest_xray")
    parser.add_argument("--split_csv", default="dataset/split.csv")
    parser.add_argument("--checkpoint", default="models/best_pneumonia_resnet18.pth")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--min_sensitivity", type=float, default=0.95,
                         help="Minimum pneumonia recall required when tuning the "
                              "threshold on validation (default: 0.95)")
    parser.add_argument("--uncertain_band", type=float, nargs=2, default=None,
                         metavar=("LOWER", "UPPER"),
                         help="Optional: report an 'uncertain, refer' coverage/accuracy "
                              "table for this probability band on the test set, "
                              "e.g. --uncertain_band 0.4 0.6")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.results_dir, exist_ok=True)

    print("Loading data...")
    _, val_loader, test_loader, _ = get_dataloaders(
        dataset_dir=args.dataset_dir,
        split_csv=args.split_csv,
        batch_size=args.batch_size,
    )
    print(f"Validation images: {len(val_loader.dataset)}, Test images: {len(test_loader.dataset)}")

    print(f"Loading {args.model} model from {args.checkpoint} ...")
    model = load_checkpoint_for_model(args.model, args.checkpoint, device)

    print("Running inference on VALIDATION set...")
    y_true_val, y_prob_val = get_probs_and_labels(model, val_loader, device)

    print("Running inference on TEST set...")
    y_true_test, y_prob_test = get_probs_and_labels(model, test_loader, device)

    # --- Default threshold (0.5) ---
    val_metrics_default = compute_metrics(y_true_val, y_prob_val, 0.5)
    test_metrics_default = compute_metrics(y_true_test, y_prob_test, 0.5)

    # --- Tuned threshold: chosen on VALIDATION ONLY, applied once to TEST ---
    tuned_threshold, reached_target = tune_threshold(y_true_val, y_prob_val, args.min_sensitivity)
    val_metrics_tuned = compute_metrics(y_true_val, y_prob_val, tuned_threshold)
    test_metrics_tuned = compute_metrics(y_true_test, y_prob_test, tuned_threshold)

    results = {
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "min_sensitivity_target": args.min_sensitivity,
        "tuned_threshold": tuned_threshold,
        "tuned_threshold_reached_target_on_val": reached_target,
        "validation": {
            "default_threshold_0.5": val_metrics_default,
            "tuned_threshold": val_metrics_tuned,
        },
        "test": {
            "default_threshold_0.5": test_metrics_default,
            "tuned_threshold": test_metrics_tuned,
        },
    }

    if args.uncertain_band is not None:
        lower, upper = args.uncertain_band
        results["test"]["uncertain_band"] = uncertain_band_table(y_true_test, y_prob_test, lower, upper)

    metrics_path = os.path.join(args.results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

    plot_confusion_matrix(
        test_metrics_default["confusion_matrix"],
        "Test Confusion Matrix (threshold=0.5)",
        os.path.join(args.results_dir, "confusion_matrix_test_default.png"),
    )
    plot_confusion_matrix(
        test_metrics_tuned["confusion_matrix"],
        f"Test Confusion Matrix (tuned threshold={tuned_threshold:.3f})",
        os.path.join(args.results_dir, "confusion_matrix_test_tuned.png"),
    )
    plot_roc_curve(
        y_true_test, y_prob_test, test_metrics_default["roc_auc"],
        "Test ROC Curve", os.path.join(args.results_dir, "roc_curve_test.png"),
    )
    print(f"Saved plots to {args.results_dir}/")

    print("\n===== SUMMARY =====")
    print(f"Tuned threshold (chosen on val, target sensitivity >= {args.min_sensitivity}): "
          f"{tuned_threshold:.4f} (target reached: {reached_target})")
    print("\n--- TEST @ 0.5 ---")
    print(f"Accuracy: {test_metrics_default['accuracy']:.4f}  ROC-AUC: {test_metrics_default['roc_auc']:.4f}")
    print(f"Pneumonia recall (sensitivity): {test_metrics_default['sensitivity_pneumonia_recall']:.4f}")
    print(f"Normal recall (specificity): {test_metrics_default['specificity_normal_recall']:.4f}")
    print("\n--- TEST @ tuned threshold ---")
    print(f"Accuracy: {test_metrics_tuned['accuracy']:.4f}  ROC-AUC: {test_metrics_tuned['roc_auc']:.4f}")
    print(f"Pneumonia recall (sensitivity): {test_metrics_tuned['sensitivity_pneumonia_recall']:.4f}")
    print(f"Normal recall (specificity): {test_metrics_tuned['specificity_normal_recall']:.4f}")


if __name__ == "__main__":
    main()
