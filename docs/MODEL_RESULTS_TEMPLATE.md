# CivicLens model experiment report (fill only after training)

- Run date:
- Dataset version / DOI:
- Dataset licence / attribution:
- Data counts per class:
- Train/validation split:
- Model: MobileNetV3-Small, ImageNet transfer learning
- Training epochs / batch size / learning rate:

## Validation results

| Task | Validation accuracy | Macro-F1 | Confusion matrix |
|---|---:|---:|---|
| Pothole vs non-pothole | Fill from `models/pothole_metrics.json` | Fill from file | Fill from file |
| Garbage vs non-garbage | Fill from `models/garbage_metrics.json` | Fill from file | Fill from file |

## Qualitative error analysis

Test several held-out photos from different lighting, camera angles, road types, and backgrounds. Record false positives and false negatives. Do not use the training images as evidence of generalization.

## Limitations

The classifier only covers pothole and garbage tasks. Validation results on this dataset do not establish accuracy for every city, phone camera, weather condition, or civic issue. Predictions must be checked by a person. CivicLens does not submit reports to government portals.
