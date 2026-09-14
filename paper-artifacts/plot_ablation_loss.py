import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs('paper-artifacts/figs', exist_ok=True)

experiments = [
    ('exp1', 'Wide-ResNet-101-2 (Exp. 1)'),
    ('exp2', 'Vision Transformer (Exp. 2)'),
    ('exp3', 'ResNet-18 (Exp. 3)')
]

variants = [
    ('adam', 'Adam (Baseline)', '#7f7f7f', '--', 1.5),
    ('trinity_full', 'Trinity (Full)', '#1f77b4', '-', 2.2),
    ('without_kfac', 'w/o K-FAC (SO off)', '#ff7f0e', '-.', 1.6),
    ('kfac_always_on', 'Always-On K-FAC', '#9467bd', '-', 1.6),
    ('no_grafting', 'w/o Magnitude Grafting', '#d62728', ':', 1.8),
    ('no_gc', 'w/o Grad. Centralization', '#8c564b', (0, (3, 1, 1, 1)), 1.6),
    ('no_escape', 'w/o Saddle Escape', '#17becf', (0, (5, 2)), 1.6)
]

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.5,
    'figure.titlesize': 14
})

fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharex=True)

for ax, (exp_id, title) in zip(axes, experiments):
    for var_id, label, color, style, lw in variants:
        csv_file = f"paper-artifacts/CSVs_ablation/{exp_id}_{var_id}.csv"
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            ax.plot(df['epoch'], df['loss'], label=label, color=color, linestyle=style, linewidth=lw, alpha=0.9)
    
    ax.set_yscale('log')
    ax.set_title(title, fontweight='bold', pad=10)
    ax.set_xlabel('Epoch', fontweight='semibold')
    ax.set_ylabel('Training Loss (log scale)', fontweight='semibold')
    ax.grid(True, which='both', linestyle='--', alpha=0.4, linewidth=0.6)
    ax.set_xlim(1, 200)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=4, frameon=True, fancybox=True, shadow=False)

plt.tight_layout()
png_path = 'paper-artifacts/figs/ablation_loss_comparison.png'
pdf_path = 'paper-artifacts/figs/ablation_loss_comparison.pdf'
plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.savefig(pdf_path, bbox_inches='tight')
plt.close()

print(f"Generated {png_path} and {pdf_path}")
