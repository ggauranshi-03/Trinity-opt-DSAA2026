"""
Plot val-accuracy-vs-epoch curves for all optimizers on a given model/dataset,
reading the CSVs written by train_resnet18_dsaa.py / train_transformer_dsaa.py /
train_wide_resnet_dsaa.py (one CSV per --optimizer run).

Usage:
    python plot_optimizer_comparison.py --model resnet18 --dataset cifar10 --imb_factor 0.01
    python plot_optimizer_comparison.py --model all --output_dir results_dsaa2026 --save_dir figs
"""
import os
import csv
import argparse

import matplotlib.pyplot as plt

MODEL_PREFIX = {
    'resnet18': 'resnet18',
    'wideresnet': 'wideresnet',
    'transformer': 'transformer',
}
DEFAULT_OPTIMIZERS = ['adam', 'rmsprop', 'sgd', 'trinity']
COLORS = {
    'adam': '#1f77b4',
    'rmsprop': '#ff7f0e',
    'sgd': '#2ca02c',
    'trinity': '#d62728',
}
DISPLAY_NAME = {
    'adam': 'Adam',
    'rmsprop': 'RMSprop',
    'sgd': 'SGD',
    'trinity': 'Trinity',
}


def read_csv(path, metric):
    epochs, values = [], []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            values.append(float(row[metric]))
    return epochs, values


def plot_model(model, dataset, imb_factor, output_dir, optimizers, metric, save_dir):
    prefix = MODEL_PREFIX[model]
    fig, ax = plt.subplots(figsize=(7, 4.5))

    curves = {}
    for opt_name in optimizers:
        csv_path = os.path.join(output_dir, f"{prefix}_{opt_name}_{dataset}_imb{imb_factor}.csv")
        if not os.path.exists(csv_path):
            print(f"[skip] missing {csv_path}")
            continue
        epochs, values = read_csv(csv_path, metric)
        if not epochs:
            continue
        curves[opt_name] = (epochs, values)
        ax.plot(epochs, values, label=DISPLAY_NAME.get(opt_name, opt_name),
                color=COLORS.get(opt_name), linewidth=1.6)

    # annotate the final value of each curve just past the right edge, staggering
    # labels that would otherwise collide (closely-converging curves, as in Fig. seen
    # in the paper's accuracy plots)
    if curves:
        x_max = max(epochs[-1] for epochs, _ in curves.values())
        y_min = min(min(v) for _, v in curves.values())
        y_max = max(max(v) for _, v in curves.values())
        y_span = max(y_max - y_min, 1e-6)
        ax.set_xlim(right=x_max + 0.14 * max(x_max, 1))

        final_vals = sorted(
            ((opt_name, epochs[-1], values[-1]) for opt_name, (epochs, values) in curves.items()),
            key=lambda t: t[2])
        # spread stacked labels evenly around their true values so text never overlaps
        n = len(final_vals)
        min_gap = 0.05 * y_span
        label_ys = [y for _, _, y in final_vals]
        for i in range(1, n):
            if label_ys[i] - label_ys[i - 1] < min_gap:
                label_ys[i] = label_ys[i - 1] + min_gap

        for (opt_name, x_end, y_end), label_y in zip(final_vals, label_ys):
            ax.annotate(f"{y_end:.2f}%",
                        xy=(x_end, y_end),
                        xytext=(x_end + 0.05 * max(x_max, 1), label_y),
                        textcoords='data', annotation_clip=False,
                        color=COLORS.get(opt_name, 'black'), fontsize=9, fontweight='bold',
                        va='center', ha='left',
                        arrowprops=dict(arrowstyle='-', linestyle='--',
                                         color=COLORS.get(opt_name, 'black'), lw=0.8))

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy (%)' if metric == 'val_acc' else 'Training Accuracy (%)')
    ax.set_title(f"{model} — {dataset} (IF via imb_factor={imb_factor})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{prefix}_{dataset}_imb{imb_factor}_{metric}.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='all',
                        choices=['resnet18', 'wideresnet', 'transformer', 'all'])
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--imb_factor', type=float, default=0.01)
    parser.add_argument('--output_dir', type=str, default='results_dsaa2026',
                        help='directory containing the per-optimizer training CSVs')
    parser.add_argument('--save_dir', type=str, default='figs',
                        help='directory to write the comparison plots to')
    parser.add_argument('--optimizers', type=str, nargs='+', default=DEFAULT_OPTIMIZERS)
    parser.add_argument('--metric', type=str, default='val_acc', choices=['val_acc', 'train_acc'])
    args = parser.parse_args()

    models = list(MODEL_PREFIX.keys()) if args.model == 'all' else [args.model]
    for model in models:
        plot_model(model, args.dataset, args.imb_factor, args.output_dir,
                   args.optimizers, args.metric, args.save_dir)


if __name__ == '__main__':
    main()
