"""Figure S3: Save each 3x6 Ramachandran panel as a separate figure (no labels).

18 panels: 3 conditions (PED ref, PED+CODLAD, CG+CODLAD) x 6 classes.
"""
import os, sys, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT = '/MDdata/data04/jxhuang/cg_cascade'
OUT = os.path.join(PROJECT, 'logs/figure3_step100')
PANEL_DIR = os.path.join(OUT, 'panels_rama')
os.makedirs(PANEL_DIR, exist_ok=True)

RESIDUE_CLASSES = ['General', 'Glycine', 'Ile/Val', 'Pre-Pro', 'Trans-Pro', 'Cis-Pro']
MAX_POINTS = 5000
np.random.seed(42)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 18, "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14,
    "figure.dpi": 600, "savefig.dpi": 600,
})


def save_panel(fig, name):
    fig.savefig(os.path.join(PANEL_DIR, f'rama_{name}.png'), dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(PANEL_DIR, f'rama_{name}.pdf'), dpi=600, bbox_inches='tight')
    plt.close(fig)


def panel_one(ax, sub, cond_label):
    n_total = len(sub)
    if n_total > MAX_POINTS:
        idx = np.random.choice(n_total, MAX_POINTS, replace=False)
        sub_plot = sub.iloc[idx]
    else:
        sub_plot = sub

    favored = sub_plot[sub_plot['classification'] == 'favored']
    allowed = sub_plot[sub_plot['classification'] == 'allowed']
    outlier = sub_plot[sub_plot['classification'] == 'outlier']

    if len(favored) > 0:
        ax.scatter(favored['phi'], favored['psi'], s=6, c='#333333', alpha=0.7, rasterized=True)
    if len(allowed) > 0:
        ax.scatter(allowed['phi'], allowed['psi'], s=5, c='#999999', alpha=0.6, rasterized=True)
    if len(outlier) > 0:
        ax.scatter(outlier['phi'], outlier['psi'], s=8, c='red', alpha=0.8,
                  edgecolors='darkred', linewidths=0.3, rasterized=True)

    n_out = (sub['classification'] == 'outlier').sum()
    pct_out = 100.0 * n_out / n_total if n_total > 0 else 0
    ax.text(0.03, 0.97, f'outliers: {pct_out:.1f}%', transform=ax.transAxes,
            fontsize=9, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
    ax.set_xticks([-90, 0, 90]); ax.set_yticks([-90, 0, 90])
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_xlabel(r'$\phi$ (°)'); ax.set_ylabel(r'$\psi$ (°)')


def main():
    cache_path = os.path.join(OUT, 'rama_S3_points.npz')
    cached = np.load(cache_path, allow_pickle=True)
    df_gt = pd.DataFrame(list(cached['gt']))
    df_pred = pd.DataFrame(list(cached['pred']))
    df_cg = pd.DataFrame(list(cached['cg']))
    print(f'Loaded: GT={len(df_gt)}, Pred={len(df_pred)}, CG={len(df_cg)}')

    conditions = [
        ('PED_ref', 'PED reference', df_gt),
        ('PED_CODLAD', 'PED+CODLAD', df_pred),
        ('CG_CODLAD', 'CG+CODLAD', df_cg),
    ]

    for cond_key, cond_name, df in conditions:
        for cls in RESIDUE_CLASSES:
            sub = df[df['category'] == cls]
            if len(sub) == 0:
                continue
            fig, ax = plt.subplots(figsize=(6, 5.5))
            panel_one(ax, sub, cond_name)
            ax.set_title(f'{cond_name} — {cls}', fontsize=14, fontweight='bold')
            name = f'{cond_key}_{cls.replace("/", "_")}'
            save_panel(fig, name)
            print(f'  {name} ({len(sub)} pts)')

    print(f'\nAll panels saved to {PANEL_DIR}/')


if __name__ == '__main__':
    main()
