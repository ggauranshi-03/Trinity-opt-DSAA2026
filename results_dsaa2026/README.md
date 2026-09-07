# Experimental Results (DSAA 2026)

This directory contains the official experimental results and benchmark CSV logs for the Trinity optimizer submitted to DSAA 2026.

## CSV File Naming Convention

- `resnet18_<dataset>_imb<factor>.csv`: Full training trajectory for ResNet-18 benchmark.
- `wideresnet_<dataset>_imb<factor>.csv`: Full training trajectory for Wide-ResNet-101-2 benchmark.
- `transformer_<dataset>_imb<factor>.csv`: Full training trajectory for 100M Transformer benchmark.
- `ablation_<variant>_<dataset>_imb<factor>.csv`: ResNet-18 ablation across component combinations (`FO`, `ZO`, `SO`, `FO_ZO`, `FO_SO`, `Full`).
- `transformer_<variant>_<dataset>_imb<factor>.csv`: Transformer ablation study across component combinations.

## Metrics Schema

Each CSV record contains the following epoch-level performance statistics:

| Column | Description |
| :--- | :--- |
| `epoch` | Epoch index (0-indexed or 1-indexed depending on run) |
| `train_loss` | Cross-entropy loss on training set |
| `train_acc` | Training accuracy (%) |
| `val_loss` | Cross-entropy loss on test set |
| `val_acc` | Overall validation/test accuracy (%) |
| `head_acc` | Accuracy on Many-Shot classes (>100 training samples) |
| `med_acc` | Accuracy on Medium-Shot classes (20 to 100 training samples) |
| `tail_acc` | Accuracy on Few-Shot / Tail classes (<20 training samples) |
| `time_s` | Cumulative runtime in seconds |
