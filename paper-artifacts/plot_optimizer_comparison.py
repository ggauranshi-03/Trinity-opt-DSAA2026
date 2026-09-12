"""
Plot training loss and validation accuracy comparison curves for all optimizers
across the three experimental paradigms:
  - Exp 1: Wide-ResNet-101-2 (exp1_*.csv)
  - Exp 2: Vision Transformer (exp2_*.csv)
  - Exp 3: ResNet-18 (exp3_*.csv)

Supports generating individual high-resolution PNG & PDF figures, as well as a
comprehensive 3x2 comparison grid figure.
"""
import os
import csv
import argparse
import matplotlib.pyplot as plt

EXPERIMENTS = {
    'exp1': {
        'title': 'Exp 1: Wide-ResNet-101-2',
        'prefix': 'exp1',
        'files': {
            'trinity': 'exp1_hybrid.csv',
            'adam': 'exp1_adam.csv',
            'rmsprop': 'exp1_rmsprop.csv',
            'sgd': 'exp1_sgd.csv'
        }
    },
    'exp2': {
        'title': 'Exp 2: Vision Transformer',
        'prefix': 'exp2',
        'files': {
            'trinity': 'exp2_hybrid.csv',
            'adam': 'exp2_adam.csv',
            'rmsprop': 'exp2_rmsprop.csv',
            'sgd': 'exp2_sgd.csv'
        }
    },
    'exp3': {
        'title': 'Exp 3: ResNet-18',
        'prefix': 'exp3',
        'files': {
            'trinity': 'exp3_hybrid.csv',
            'adam': 'exp3_adam.csv',
            'rmsprop': 'exp3_rmsprop.csv',
            'sgd': 'exp3_sgd.csv'
        }
    }
}

COLORS = {
    'trinity': '#1f77b4',  # Steel Blue (matches colorhybrid in LaTeX)
    'adam': '#d62728',     # Muted Red (matches coloradam in LaTeX)
    'rmsprop': '#2ca02c',  # Forest Green (matches colorrmsprop in LaTeX)
    'sgd': '#ff7f0e',      # Safety Orange (matches colorsgd in LaTeX)
}

DISPLAY_NAMES = {
    'trinity': 'Trinity (Proposed)',
    'adam': 'Adam',
    'rmsprop': 'RMSProp',
    'sgd': 'SGD (Nesterov)',
}

def read_metric_csv(path, metric):
    epochs, values = [], []
    if not os.path.exists(path):
        return epochs, values
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            values.append(float(row[metric]))
    return epochs, values

def plot_single(exp_key, metric, save_dir='figs'):
    info = EXPERIMENTS[exp_key]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    
    is_loss = (metric == 'loss')
    curves = {}
    
    for opt in ['trinity', 'adam', 'rmsprop', 'sgd']:
        csv_file = info['files'][opt]
        epochs, values = read_metric_csv(csv_file, metric)
        if epochs:
            curves[opt] = (epochs, values)
            lw = 2.0 if opt == 'trinity' else 1.4
            ax.plot(epochs, values, label=DISPLAY_NAMES[opt],
                    color=COLORS[opt], linewidth=lw)
    
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    if is_loss:
        ax.set_ylabel('Training Loss (log scale)', fontsize=11, fontweight='bold')
        ax.set_yscale('log')
        ax.set_title(f"{info['title']} — Training Loss", fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.95)
    else:
        ax.set_ylabel('Validation Accuracy (%)', fontsize=11, fontweight='bold')
        ax.set_title(f"{info['title']} — Validation Accuracy", fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', framealpha=0.95)
        
    ax.grid(True, which='both' if is_loss else 'major', alpha=0.3)
    fig.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    out_base = os.path.join(save_dir, f"{exp_key}_{metric}")
    fig.savefig(f"{out_base}.png", dpi=250)
    fig.savefig(f"{out_base}.pdf")
    plt.close(fig)
    print(f"[saved] {out_base}.png and .pdf")

def plot_combined_grid(save_dir='figs'):
    """Generates a 3x2 grid figure: Rows = Exp 1, 2, 3; Cols = Loss, Accuracy."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    
    exp_keys = ['exp1', 'exp2', 'exp3']
    metrics = ['loss', 'val_acc']
    
    for row_idx, exp_key in enumerate(exp_keys):
        info = EXPERIMENTS[exp_key]
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            is_loss = (metric == 'loss')
            
            for opt in ['trinity', 'adam', 'rmsprop', 'sgd']:
                csv_file = info['files'][opt]
                epochs, values = read_metric_csv(csv_file, metric)
                if epochs:
                    lw = 2.0 if opt == 'trinity' else 1.3
                    ax.plot(epochs, values, label=DISPLAY_NAMES[opt],
                            color=COLORS[opt], linewidth=lw)
            
            ax.set_xlabel('Epoch', fontsize=10)
            if is_loss:
                ax.set_ylabel('Training Loss', fontsize=10)
                ax.set_yscale('log')
                ax.set_title(f"{info['title']} — Loss", fontsize=11, fontweight='bold')
                ax.legend(loc='upper right', fontsize=8.5, framealpha=0.9)
            else:
                ax.set_ylabel('Validation Accuracy (%)', fontsize=10)
                ax.set_title(f"{info['title']} — Val Accuracy", fontsize=11, fontweight='bold')
                ax.legend(loc='lower right', fontsize=8.5, framealpha=0.9)
            ax.grid(True, which='both' if is_loss else 'major', alpha=0.25)
            
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    grid_out = os.path.join(save_dir, "all_experiments_grid")
    fig.savefig(f"{grid_out}.png", dpi=250)
    fig.savefig(f"{grid_out}.pdf")
    plt.close(fig)
    print(f"[saved] {grid_out}.png and .pdf")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='figs')
    args = parser.parse_args()
    
    for exp_key in ['exp1', 'exp2', 'exp3']:
        plot_single(exp_key, 'loss', args.save_dir)
        plot_single(exp_key, 'val_acc', args.save_dir)
    plot_combined_grid(args.save_dir)

if __name__ == '__main__':
    main()
