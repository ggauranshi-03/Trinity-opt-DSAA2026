import os
import re
import csv
import pandas as pd

output_dir = 'paper-artifacts/CSVs_ablation'
os.makedirs(output_dir, exist_ok=True)

pattern = re.compile(r'Epoch (\d+)/\d+.*?Loss:\s*([0-9.]+).*?Val Acc:\s*([0-9.]+)%')

def parse_log_segment(lines, out_csv):
    rows = []
    for line in lines:
        m = pattern.search(line)
        if m:
            ep = int(m.group(1))
            ls = float(m.group(2))
            va = float(m.group(3))
            rows.append((ep, ls, va))
    if rows:
        with open(out_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'loss', 'val_acc'])
            for r in rows:
                writer.writerow(r)
        print(f"Generated {out_csv} with {len(rows)} epochs. Final loss={rows[-1][1]}, val_acc={rows[-1][2]}%")
    else:
        print(f"Warning: No rows found for {out_csv}")

# ==============================================================================
# EXP 1: Wide-ResNet-101-2
# ==============================================================================
exp1_sources = {
    'adam': 'ablation_res/wideresnet_adam_cifar10_imb0.01.csv',
    'trinity_full': 'wandb/run-20260909_152957-r0dnvqv8/files/output.log',
    'without_kfac': 'wandb/run-20260909_180954-br1ettwe/files/output.log',
    'kfac_always_on': 'wandb/run-20260909_124257-m8kevo0l/files/output.log',
    'no_grafting': 'wandb/run-20260909_172747-ii3hp569/files/output.log',
    'no_gc': 'wandb/run-20260909_164722-nnhkabd4/files/output.log',
    'no_escape': 'wandb/run-20260909_160838-tn3w5nwj/files/output.log',
}

for name, src in exp1_sources.items():
    out_csv = os.path.join(output_dir, f"exp1_{name}.csv")
    if src.endswith('.csv'):
        df = pd.read_csv(src)
        df[['epoch', 'train_loss', 'val_acc']].rename(columns={'train_loss': 'loss'}).to_csv(out_csv, index=False)
        print(f"Copied {out_csv} from {src}")
    else:
        with open(src, 'r') as f:
            lines = f.readlines()
        parse_log_segment(lines, out_csv)

# ==============================================================================
# EXP 2: Vision Transformer
# ==============================================================================
exp2_sources = {
    'adam': 'ablation_res/transformer_small_adam_cifar10_imb0.01.csv',
    'trinity_full': 'ablation_res/transformer_small_trinity_full_cifar10_imb0.01.csv',
    'without_kfac': 'ablation_res/transformer_small_trinity_without_kfac_cifar10_imb0.01.csv',
    'kfac_always_on': 'ablation_res/transformer_small_kfac_always_on_cifar10_imb0.01.csv',
    'no_grafting': 'ablation_res/transformer_small_trinity_no_grafting_cifar10_imb0.01.csv',
    'no_gc': 'ablation_res/transformer_small_trinity_no_gc_cifar10_imb0.01.csv',
    'no_escape': 'ablation_res/transformer_small_trinity_no_escape_cifar10_imb0.01.csv',
}

for name, src in exp2_sources.items():
    out_csv = os.path.join(output_dir, f"exp2_{name}.csv")
    df = pd.read_csv(src)
    df[['epoch', 'train_loss', 'val_acc']].rename(columns={'train_loss': 'loss'}).to_csv(out_csv, index=False)
    print(f"Copied {out_csv} from {src}")

# ==============================================================================
# EXP 3: ResNet-18 (Extracted from logs/ablation_suite.log)
# ==============================================================================
suite_log = 'logs/ablation_suite.log'
if os.path.exists(suite_log):
    with open(suite_log, 'r') as f:
        content = f.read()

    # Split on wandb run setup
    runs = content.split('wandb: setting up run ')
    # runs[0] is pre-run
    # runs[1]: adam
    # runs[2]: kfac_always_on
    # runs[3]: trinity_full
    # runs[4]: trinity_no_escape
    # runs[5]: trinity_no_gc
    # runs[6]: trinity_no_grafting
    # runs[7]: trinity_without_kfac
    run_names = [
        ('adam', 1),
        ('kfac_always_on', 2),
        ('trinity_full', 3),
        ('no_escape', 4),
        ('no_gc', 5),
        ('no_grafting', 6),
        ('without_kfac', 7),
    ]

    for name, idx in run_names:
        if idx < len(runs):
            out_csv = os.path.join(output_dir, f"exp3_{name}.csv")
            lines = runs[idx].splitlines()
            parse_log_segment(lines, out_csv)
