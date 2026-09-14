import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load dataset
csv_path = 'paper-artifacts/CSVs_ablation/trinity_kfac_dynamics.csv'
df = pd.read_csv(csv_path)

# Academic styling for DSAA / IEEE
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10.5
plt.rcParams['axes.labelsize'] = 11.5
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10.5
plt.rcParams['ytick.labelsize'] = 10.5
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 13.5

c_wrn = '#1f77b4'   # Steel Blue
c_vit = '#9467bd'   # Purple
c_res = '#2ca02c'   # Forest Green

# =========================================================================
# Figure 1: Multi-Architecture K-FAC Allocation % and Mean Rho Dynamics
# =========================================================================
fig, ax1 = plt.subplots(figsize=(11.0, 5.4), dpi=300)
fig.subplots_adjust(left=0.08, right=0.91, top=0.88, bottom=0.12)

ax2 = ax1.twinx()

# --- Left Axis: K-FAC Allocation % (Solid Lines) ---
l1, = ax1.plot(df['epoch'], df['wrn_so_pct'], color=c_wrn, linewidth=2.5, linestyle='-', label='Wide-ResNet-101-2 (Cap: 38.1%, 40/105 blks)')
l2, = ax1.plot(df['epoch'], df['res_so_pct'], color=c_res, linewidth=2.5, linestyle='-', label='ResNet-18 (Cap: 28.6%, 6/21 blks)')
l3, = ax1.plot(df['epoch'], df['vit_so_pct'], color=c_vit, linewidth=2.5, linestyle='-', label='Vision Transformer (Cap: 21.1%, 8/38 blks)')

# Dashed budget saturation lines
ax1.axhline(38.10, color=c_wrn, linestyle=':', alpha=0.55, linewidth=1.1)
ax1.axhline(28.57, color=c_res, linestyle=':', alpha=0.55, linewidth=1.1)
ax1.axhline(21.05, color=c_vit, linestyle=':', alpha=0.55, linewidth=1.1)

ax1.set_xlim(1, 200)
ax1.set_ylim(-1, 54)
ax1.set_xlabel('Training Epoch', fontweight='bold')
ax1.set_ylabel('Active K-FAC Blocks (% of Eligible Blocks)', fontweight='bold', color='#111111')
ax1.tick_params(axis='y', labelcolor='#111111')
ax1.grid(True, linestyle='--', alpha=0.45)

# --- Right Axis: Mean Rho Values (Dashed Lines) ---
r1, = ax2.plot(df['epoch'], df['wrn_rho'], color=c_wrn, linewidth=2.0, linestyle='--', alpha=0.85, label=r'Wide-ResNet-101-2 (Mean $\bar{\rho}$)')
r2, = ax2.plot(df['epoch'], df['res_rho'], color=c_res, linewidth=2.0, linestyle='--', alpha=0.85, label=r'ResNet-18 (Mean $\bar{\rho}$)')
r3, = ax2.plot(df['epoch'], df['vit_rho'], color=c_vit, linewidth=2.0, linestyle='--', alpha=0.85, label=r'Vision Transformer (Mean $\bar{\rho}$)')

# Threshold reference line (exact hyperparameter from trinity_optimizer.py)
r_th = ax2.axhline(0.15, color='#666666', linestyle='dashdot', linewidth=1.2, alpha=0.8, label=r'Hysteresis Trigger ($\rho_{\mathrm{lo}} = 0.15$)')
ax2.axhspan(0.15, 0.35, color='#888888', alpha=0.08, label='_nolegend_')

ax2.set_ylim(-0.02, 1.05)
ax2.set_ylabel(r'Curvature Anisotropy Ratio $\bar{\rho}$ ($r_{\mathrm{eff}} / d$)', fontweight='bold', color='#333333')
ax2.tick_params(axis='y', labelcolor='#333333')
ax2.grid(False)

# Shaded background warmup
ax1.axvspan(1, 8, color='#f0f4f8', alpha=0.85)
ax1.text(4.5, 8.5, 'Pure FO\n(AdamW)', ha='center', va='center', fontsize=8.2, color='#444444', fontweight='bold', style='italic',
         bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#cccccc', alpha=0.9))

# Callouts for trigger epochs
ax1.annotate('ViT Trigger\n(Epoch 8)', xy=(8, 5.26), xytext=(20, 11),
            arrowprops=dict(arrowstyle='->', color=c_vit, lw=1.3),
            fontsize=9, fontweight='bold', color=c_vit, ha='left')

ax1.annotate('WRN Trigger\n(Epoch 33)', xy=(33, 6.67), xytext=(45, 12),
            arrowprops=dict(arrowstyle='->', color=c_wrn, lw=1.3),
            fontsize=9, fontweight='bold', color=c_wrn, ha='left')

ax1.annotate('ResNet Trigger\n(Epoch 55)', xy=(55, 4.76), xytext=(68, 6),
            arrowprops=dict(arrowstyle='->', color=c_res, lw=1.3),
            fontsize=9, fontweight='bold', color=c_res, ha='left')

# Detailed 2-column legend
lines = [l1, r1, l2, r2, l3, r3, r_th]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right', bbox_to_anchor=(0.985, 0.98),
           ncol=2, frameon=True, framealpha=0.95, edgecolor='#cccccc', fontsize=8.4,
           columnspacing=1.2, handletextpad=0.5)

ax1.set_title(r'Dynamic K-FAC Preconditioning Engagement & Curvature Anisotropy (Mean $\bar{\rho}$)', fontweight='bold', pad=12)

fig.savefig('paper-artifacts/figs/kfac_engagement_percentage.png')
fig.savefig('paper-artifacts/figs/kfac_engagement_percentage.pdf')
plt.close(fig)
print('Figure 1 created: paper-artifacts/figs/kfac_engagement_percentage.png')

# =========================================================================
# Figure 2: Dual Panel (1 Row, 2 Columns with Top Legend, No Hysteresis Clutter)
# =========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300)
fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.14, wspace=0.25)

# --- Left Panel: K-FAC Allocation % ---
l1, = ax1.plot(df['epoch'], df['wrn_so_pct'], color=c_wrn, linewidth=2.3, label='Wide-ResNet-101-2 (Cap: 38.1%)')
l2, = ax1.plot(df['epoch'], df['res_so_pct'], color=c_res, linewidth=2.3, label='ResNet-18 (Cap: 28.6%)')
l3, = ax1.plot(df['epoch'], df['vit_so_pct'], color=c_vit, linewidth=2.3, label='Vision Transformer (Cap: 21.1%)')

# Dotted budget caps
ax1.axhline(38.10, color=c_wrn, linestyle=':', alpha=0.6, linewidth=1.2)
ax1.axhline(28.57, color=c_res, linestyle=':', alpha=0.6, linewidth=1.2)
ax1.axhline(21.05, color=c_vit, linestyle=':', alpha=0.6, linewidth=1.2)

ax1.set_xlim(1, 200)
ax1.set_ylim(-1, 46)
ax1.set_xlabel('Training Epoch', fontweight='bold', fontsize=10)
ax1.set_ylabel('Active K-FAC Blocks (%)', fontweight='bold', fontsize=10)
ax1.set_title('(a) Second-Order Allocation (%)', fontweight='bold', fontsize=11, pad=8)
ax1.grid(True, linestyle='--', alpha=0.5)

# --- Right Panel: Curvature Anisotropy Ratio (rho) ---
ax2.plot(df['epoch'], df['wrn_rho'], color=c_wrn, linewidth=2.3)
ax2.plot(df['epoch'], df['res_rho'], color=c_res, linewidth=2.3)
ax2.plot(df['epoch'], df['vit_rho'], color=c_vit, linewidth=2.3)

ax2.set_xlim(1, 200)
ax2.set_ylim(-0.02, 1.05)
ax2.set_xlabel('Training Epoch', fontweight='bold', fontsize=10)
ax2.set_ylabel(r'Curvature Anisotropy Ratio $\bar{\rho}$', fontweight='bold', fontsize=10)
ax2.set_title(r'(b) Spectral Anisotropy Evolution ($\bar{\rho}$)', fontweight='bold', fontsize=11, pad=8)
ax2.grid(True, linestyle='--', alpha=0.5)

# --- Shared Legend Above Both Plots ---
fig.legend([l1, l2, l3],
           ['Wide-ResNet-101-2 (Cap: 38.1%, 40/105 blks)',
            'ResNet-18 (Cap: 28.6%, 6/21 blks)',
            'Vision Transformer (Cap: 21.1%, 8/38 blks)'],
           loc='upper center', bbox_to_anchor=(0.53, 0.98),
           ncol=3, frameon=True, framealpha=0.95, edgecolor='#cccccc', fontsize=9.5)

fig.savefig('paper-artifacts/figs/kfac_dynamics_dual_panel.png')
fig.savefig('paper-artifacts/figs/kfac_dynamics_dual_panel.pdf')
plt.close(fig)
print('Figure 2 created: paper-artifacts/figs/kfac_dynamics_dual_panel.png')

# =========================================================================
# Figure 2b: Dedicated Single-Column (3.45 in) 1-Row 2-Panel Figure
# =========================================================================
fig_sc, (ax1_sc, ax2_sc) = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=300)
fig_sc.subplots_adjust(left=0.10, right=0.98, top=0.74, bottom=0.16, wspace=0.28)

# (a) Allocation
l1_sc, = ax1_sc.plot(df['epoch'], df['wrn_so_pct'], color=c_wrn, linewidth=2.0)
l2_sc, = ax1_sc.plot(df['epoch'], df['res_so_pct'], color=c_res, linewidth=2.0)
l3_sc, = ax1_sc.plot(df['epoch'], df['vit_so_pct'], color=c_vit, linewidth=2.0)
ax1_sc.axhline(38.10, color=c_wrn, linestyle=':', alpha=0.6, linewidth=1.1)
ax1_sc.axhline(28.57, color=c_res, linestyle=':', alpha=0.6, linewidth=1.1)
ax1_sc.axhline(21.05, color=c_vit, linestyle=':', alpha=0.6, linewidth=1.1)
ax1_sc.set_xlim(1, 200)
ax1_sc.set_ylim(-1, 46)
ax1_sc.set_xlabel('Epoch', fontweight='bold', fontsize=9.5)
ax1_sc.set_ylabel('K-FAC Blocks (%)', fontweight='bold', fontsize=9.5)
ax1_sc.set_title('(a) K-FAC Allocation (%)', fontweight='bold', fontsize=10, pad=6)
ax1_sc.tick_params(labelsize=8.5)
ax1_sc.grid(True, linestyle='--', alpha=0.5)

# (b) Rho Anisotropy
ax2_sc.plot(df['epoch'], df['wrn_rho'], color=c_wrn, linewidth=2.0)
ax2_sc.plot(df['epoch'], df['res_rho'], color=c_res, linewidth=2.0)
ax2_sc.plot(df['epoch'], df['vit_rho'], color=c_vit, linewidth=2.0)
ax2_sc.set_xlim(1, 200)
ax2_sc.set_ylim(-0.02, 1.05)
ax2_sc.set_xlabel('Epoch', fontweight='bold', fontsize=9.5)
ax2_sc.set_ylabel(r'Anisotropy Ratio $\bar{\rho}$', fontweight='bold', fontsize=9.5)
ax2_sc.set_title(r'(b) Curvature Ratio $\bar{\rho}$', fontweight='bold', fontsize=10, pad=6)
ax2_sc.tick_params(labelsize=8.5)
ax2_sc.grid(True, linestyle='--', alpha=0.5)

# Detailed Shared Legend on Top
fig_sc.legend([l1_sc, l2_sc, l3_sc],
              ['Wide-ResNet-101-2 (Cap: 38.1%, 40/105 blks)',
               'ResNet-18 (Cap: 28.6%, 6/21 blks)',
               'Vision Transformer (Cap: 21.1%, 8/38 blks)'],
              loc='upper center', bbox_to_anchor=(0.54, 0.99),
              ncol=2, frameon=True, framealpha=0.95, edgecolor='#cccccc', fontsize=8.0)

fig_sc.savefig('paper-artifacts/figs/kfac_dynamics_1col_2panel.png')
fig_sc.savefig('paper-artifacts/figs/kfac_dynamics_1col_2panel.pdf')
plt.close(fig_sc)
print('Figure 2b created: paper-artifacts/figs/kfac_dynamics_1col_2panel.png')

# =========================================================================
# Figure 3: Wide-ResNet-101-2 Dual-Axis Deep-Dive
# =========================================================================
fig, ax_left = plt.subplots(figsize=(8.5, 4.6), dpi=300)
fig.subplots_adjust(left=0.13, right=0.88, top=0.88, bottom=0.13)

ax_right = ax_left.twinx()

l1, = ax_left.plot(df['epoch'], df['wrn_so_count'], color=c_wrn, linewidth=2.6, label='Active K-FAC Blocks (Count)')
ax_left.axhline(40, color=c_wrn, linestyle=':', alpha=0.6, linewidth=1.2)
ax_left.set_ylabel('Active K-FAC Blocks ($n_{\\mathrm{SO}}$ / 105)', color=c_wrn, fontweight='bold', labelpad=8)
ax_left.tick_params(axis='y', labelcolor=c_wrn)
ax_left.set_ylim(-1, 56)

l2, = ax_right.plot(df['epoch'], df['wrn_rho'], color='#d62728', linewidth=2.2, linestyle='--', label=r'Curvature Anisotropy $\bar{\rho}$')
l3 = ax_right.axhline(0.098, color='#d62728', linestyle=':', alpha=0.6, linewidth=1.2, label=r'Threshold $\rho_{\mathrm{lo}} = 0.098$')
ax_right.set_ylabel(r'Curvature Anisotropy Ratio $\bar{\rho}$', color='#d62728', fontweight='bold', labelpad=8)
ax_right.tick_params(axis='y', labelcolor='#d62728')
ax_right.set_ylim(-0.02, 1.05)

# Highlight Phases
ax_left.axvspan(1, 32, color='#f0f4f8', alpha=0.7)
ax_left.text(3, 48, 'Phase I: Pure AdamW\n(FO Warmup, $\\bar{\\rho} > \\rho_{\\mathrm{lo}}$)',
             ha='left', va='center', fontsize=8.5, color='#444444', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#dddddd', alpha=0.9))

ax_left.annotate('Hysteresis Trigger\n(Ep 33: $\\bar{\\rho} < 0.10$)', xy=(33, 7), xytext=(48, 14),
                 arrowprops=dict(arrowstyle='->', color=c_wrn, lw=1.3),
                 fontsize=9, fontweight='bold', color=c_wrn)

ax_left.annotate('Budget Cap Bound\n($\\beta_{\\max} = 38.1\\%$, 40 blks)', xy=(36, 40), xytext=(68, 38),
                 arrowprops=dict(arrowstyle='->', color=c_wrn, lw=1.3),
                 fontsize=9, fontweight='bold', color=c_wrn)

ax_left.set_xlim(1, 200)
ax_left.set_xlabel('Training Epoch', fontweight='bold')
ax_left.set_title('Trinity K-FAC Controller Dynamics on Wide-ResNet-101-2', fontweight='bold', pad=12)

lines = [l1, l2, l3]
labels = [l.get_label() for l in lines]
ax_left.legend(lines, labels, loc='center right', frameon=True, framealpha=0.95, edgecolor='#cccccc')
ax_left.grid(True, linestyle='--', alpha=0.4)

fig.savefig('paper-artifacts/figs/kfac_engagement_wrn_dual_axis.png')
fig.savefig('paper-artifacts/figs/kfac_engagement_wrn_dual_axis.pdf')
plt.close(fig)
print('Figure 3 created: paper-artifacts/figs/kfac_engagement_wrn_dual_axis.png')
