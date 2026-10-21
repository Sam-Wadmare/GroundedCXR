# Model Comparison: ResNet18 (transfer learning) vs. SimpleCNN (from scratch)

This compares two models trained with the **same** patient-grouped split
(`dataset/split.csv`, from `src/split.py`), the **same** data augmentation
(`src/data_loader.py`'s `train_transform`), and the **same** class-weighted
cross-entropy loss (`src/train.py`'s `compute_class_weights`).

- **ResNet18**: `src/model.py`'s `create_model()` — ImageNet-pretrained,
  only `layer4` and the final `fc` layer are fine-tuned. Retrained here on
  the new patient-grouped split (the original `best_pneumonia_resnet18.pth`
  predates that split, so it is not directly comparable to SimpleCNN on
  identical data).
- **SimpleCNN**: `src/simple_cnn.py`'s `SimpleCNN()` — 4 blocks of
  Conv→BatchNorm→ReLU→MaxPool, global average pooling, one linear layer.
  No pretrained weights anywhere; every parameter starts randomly
  initialized.

## How to reproduce these numbers

```bash
python -m src.split                                              # if not already done
python -m src.train --model resnet18    --epochs 5  --output models/best_resnet18_new_split.pth
python -m src.train --model simple_cnn  --epochs 15 --output models/best_simple_cnn.pth
python -m src.evaluate --checkpoint models/best_resnet18_new_split.pth --results_dir results/resnet18_new_split
python -m src.evaluate --checkpoint models/best_simple_cnn.pth         --results_dir results/simple_cnn
```

SimpleCNN has no pretrained features to build on, so it will likely need
more epochs than ResNet18 to reach a comparable accuracy — adjust
`--epochs` as needed and record whatever you actually ran.

## Results

| Metric | ResNet18 (transfer learning) | SimpleCNN (from scratch) |
|---|---:|---:|
| Test accuracy | [PASTE RESULT] | [PASTE RESULT] |
| Pneumonia recall (sensitivity) | [PASTE RESULT] | [PASTE RESULT] |
| Normal recall (specificity) | [PASTE RESULT] | [PASTE RESULT] |
| ROC-AUC | [PASTE RESULT] | [PASTE RESULT] |
| Total parameters | 11,177,538 | 389,890 |
| Trainable parameters | 8,394,754 (layer4 + fc only; rest frozen) | 389,890 (all) |
| Training time | [PASTE RESULT] | [PASTE RESULT] |
| Epochs trained | [PASTE RESULT] | [PASTE RESULT] |
| Best validation accuracy | [PASTE RESULT] | [PASTE RESULT] |

Parameter counts above were computed directly from the instantiated
architectures (a property of the model definition, independent of
training), not measured from a training run. Everything else is a
placeholder until real training is run — fill these in from
`results/train_history_resnet18.json`, `results/train_history_simple_cnn.json`,
and the two `metrics.json` files `src/evaluate.py` produces.

## Discussion

[PASTE: a few sentences once real numbers are in — e.g. does the ~29x
larger pretrained model actually outperform the from-scratch baseline by
much, given this is a fairly distinctive binary task (lung opacity vs.
not)? Chest X-ray pneumonia classification is a task where some prior
work has found that lightweight from-scratch CNNs can get
surprisingly close to transfer-learned models — worth commenting on once
you have real numbers, with [CITATION TO VERIFY] for any such claim.]

## Limitations of this comparison

- Both models were trained with the same basic recipe (same optimizer
  type, same class weights, same augmentation) but hyperparameters
  (learning rate, epoch count, batch size) were not independently tuned
  per model — a from-scratch CNN and a fine-tuned pretrained network
  often want different learning rates, and using one fixed rate for both
  may understate SimpleCNN's true potential.
- Training time is wall-clock, highly dependent on the specific hardware
  used (CPU vs. GPU, which GPU) — only comparable between runs on the
  same machine.
- Neither model's threshold was tuned for this comparison table (both
  are reported at the default 0.5 threshold from `src/evaluate.py`,
  unless you choose to report the tuned-threshold numbers instead — be
  consistent about which one you're reporting for both models).
