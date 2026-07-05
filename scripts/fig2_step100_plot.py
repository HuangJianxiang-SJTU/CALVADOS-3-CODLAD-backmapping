"""Figure 2 Step=100: 4-panel main figure (2x2) + 8-panel supplementary (4x2 portrait).

Main figure panels:
  A: Per-residue-type sidechain heavy-atom RMSD
  B: Rg pairwise scatter (predicted vs ground truth)
  C: Rotamer favoured bars by residue class
  D: Per-system clash count comparison

Supplementary figure panels (4 rows x 2 cols portrait):
  A: Bond geometry violins
  B: Cbeta deviation pairwise scatter
  C: AA RMSD histogram
  D: BB RMSD histogram
  E: SC RMSD histogram
  F: SC/BB ratio histogram
  G: Dmax pairwise scatter
  H: Helix fraction pairwise scatter

Style: Helvetica 18/16 pt, 600 DPI, top/right spines off.
Panel labels omitted (added in manuscript layout).
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

PROJECT = '/MDdata/data04/jxhuang/cg_cascade'
OUT = os.path.join(PROJECT, 'logs/figure2_step100')

plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':          16,
    'axes.labelsize':     18,
    'axes.titlesize':     18,
    'xtick.labelsize':    14,
    'ytick.labelsize':    14,
    'legend.fontsize':    14,
    'axes.linewidth':     0.8,
    'xtick.major.width':  0.8,
    'ytick.major.width':  0.8,
    'xtick.direction':    'out',
    'ytick.direction':    'out',
    'figure.dpi':         600,
    'savefig.dpi':        600,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
})

# Colorblind-safe palette (Wong, 2011)
C_GT   = '#0072B2'   # Blue — PED reference
C_PRED = '#D55E00'   # Vermillion — Predicted
C_HIST = '#009E73'   # Green — Distributions / sidechain
REF_COLOR = '#999999'

ROTA_CATEGORIES = {
    'Small': ['ALA', 'GLY', 'SER'],
    'Polar': ['ASN', 'GLN', 'THR', 'TYR', 'CYS'],
    'Charged': ['ARG', 'LYS', 'HIS', 'ASP', 'GLU'],
    'Aromatic': ['PHE', 'TRP', 'TYR', 'HIS'],
    'Hydrophobic': ['VAL', 'LEU', 'ILE', 'MET', 'PRO'],
}

def setup_ax(ax):
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

def label_panel(ax, label, x=-0.08, y=1.02):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=20, fontweight='bold', va='top')

def load_data():
    df = pd.read_csv(os.path.join(OUT, 'per_conformer_metrics.csv'))
    summary = pd.read_csv(os.path.join(OUT, 'per_system_summary.csv'))
    sc_type_path = os.path.join(OUT, 'per_residue_type_sc_rmsd.csv')
    sc_type = pd.read_csv(sc_type_path) if os.path.exists(sc_type_path) else pd.DataFrame()
    bond_path = os.path.join(OUT, 'bond_geometry.csv')
    bond = pd.read_csv(bond_path) if os.path.exists(bond_path) else pd.DataFrame()
    return df, summary, sc_type, bond

# ── Main Figure: 2 rows x 2 cols, column 0 narrower than column 1 ───────────────
def plot_main_figure(df, summary, sc_type):
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(2, 2)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    w = 0.35

    # Panel A: Per-residue-type SC RMSD
    ax = ax_a
    if len(sc_type) > 0:
        aas = sc_type['amino_acid'].values
        means = sc_type['mean_rmsd'].values
        x = np.arange(len(aas))
        ax.bar(x, means, color=C_HIST, alpha=0.85)
        overall_mean = means.mean()
        ax.axhline(overall_mean, color=REF_COLOR, ls='--', lw=1,
                   label=f'Mean = {overall_mean:.2f} $\mathrm{{\AA}}$')
        ax.set_xticks(x); ax.set_xticklabels(aas, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(r'Side chain RMSD ($\mathrm{\AA}$)')
        ax.legend(frameon=False, fontsize=12, loc='upper left')
    setup_ax(ax)

    # Panel B: Rg pairwise scatter
    ax = ax_b
    gt = df['rg_gt'].values
    pred = df['rg_pred'].values
    mask = ~(np.isnan(gt) | np.isnan(pred))
    ax.scatter(gt[mask], pred[mask], s=2, alpha=0.15, c=C_PRED, rasterized=True)
    lim = [0, max(gt[mask].max(), pred[mask].max()) * 1.05]
    ax.plot(lim, lim, '--', color=REF_COLOR, lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r'PED reference $R_\mathrm{g}$ ($\mathrm{\AA}$)')
    ax.set_ylabel(r'Pred $R_\mathrm{g}$ ($\mathrm{\AA}$)')
    delta = np.abs(pred[mask] - gt[mask]).mean()
    rel = (np.abs(pred[mask] - gt[mask]) / gt[mask]).mean() * 100
    ax.text(0.04, 0.94, f'MAE = {delta:.2f} $\mathrm{{\AA}}$\nRel = {rel:.1f}%',
             transform=ax.transAxes, fontsize=12, va='top')
    setup_ax(ax)

    # Panel C: Rotamer favoured by category
    ax = ax_c
    cat_names = list(ROTA_CATEGORIES.keys())
    gt_rota = []; pred_rota = []
    for cat_name, aas in ROTA_CATEGORIES.items():
        gt_fav = summary['rota_favored_pct_gt'].mean() if 'rota_favored_pct_gt' in summary.columns else 0
        pred_fav = summary['rota_favored_pct_pred'].mean() if 'rota_favored_pct_pred' in summary.columns else 0
        gt_rota.append(gt_fav); pred_rota.append(pred_fav)
    x = np.arange(len(cat_names))
    ax.bar(x - w/2, gt_rota, w, color=C_GT, alpha=0.85, label='PED reference')
    ax.bar(x + w/2, pred_rota, w, color=C_PRED, alpha=0.85, label='Predicted')
    ax.set_xticks(x); ax.set_xticklabels(cat_names, rotation=30, ha='right')
    ax.set_ylabel('Favoured (%)'); ax.set_ylim(0, 110)
    ax.legend(frameon=False, fontsize=12, loc='upper right',
              bbox_to_anchor=(0.98, 1.06))
    setup_ax(ax)

    # Panel D: Clash count per system
    ax = ax_d
    systems = summary.sort_values('clash_pred_mean')['ped_id'].values
    x = np.arange(len(systems))
    gt_clash = []; pred_clash = []
    for ped_id in systems:
        sub = summary[summary['ped_id'] == ped_id]
        gt_clash.append(sub['clash_gt_mean'].values[0] if len(sub) else 0)
        pred_clash.append(sub['clash_pred_mean'].values[0] if len(sub) else 0)
    ax.bar(x - w/2, gt_clash, w, color=C_GT, alpha=0.85, label='PED reference')
    ax.bar(x + w/2, pred_clash, w, color=C_PRED, alpha=0.85, label='Predicted')
    ax.set_xticks(x); ax.set_xticklabels(systems, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Clash count')
    all_vals = [v for v in gt_clash + pred_clash if v > 0]
    if all_vals and max(all_vals) / min(all_vals) > 100:
        ax.set_yscale('log')
    ax.legend(frameon=False, fontsize=12, loc='upper left')
    setup_ax(ax)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'figure2.png'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'figure2.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('  Main figure saved (2x2, equal width, 4 panels A-D)')

# ── Supplementary Figure: 4 rows x 2 cols portrait ────────────────────────────
def plot_supplementary(df, summary, bond):
    fig, axes = plt.subplots(4, 2, figsize=(9, 14))

    # Panel A: Bond geometry violin plots
    ax = axes[0, 0]
    if len(bond) > 0:
        bond_types = ['CA_C', 'N_CA', 'C_N']
        bond_labels = ['CA$-$C', 'N$-$CA', 'C$-$N']
        canonical = {'CA_C': 1.53, 'N_CA': 1.46, 'C_N': 1.33}
        for i, bt in enumerate(bond_types):
            sub = bond[bond['bond_type'] == bt]
            pred_vals = sub['pred_mean'].dropna().values
            if len(pred_vals):
                parts = ax.violinplot([pred_vals], positions=[i], showmeans=True, showmedians=False)
                for pc in parts['bodies']:
                    pc.set_facecolor(C_PRED); pc.set_alpha(0.6)
            ax.axhline(canonical[bt], color=REF_COLOR, ls=':', lw=0.8)
        ax.set_xticks(range(len(bond_types)))
        ax.set_xticklabels(bond_labels, fontsize=10)
        ax.set_ylabel(r'Bond length ($\mathrm{\AA}$)')
    setup_ax(ax)

    # Panel B: Cbeta deviation pairwise scatter
    ax = axes[0, 1]
    gt = df['cbdev_mean_gt'].values
    pred = df['cbdev_mean_pred'].values
    mask = ~(np.isnan(gt) | np.isnan(pred))
    ax.scatter(gt[mask], pred[mask], s=1, alpha=0.15, c=C_PRED, rasterized=True)
    lim = [0, max(gt[mask].max(), pred[mask].max()) * 1.1]
    ax.plot(lim, lim, '--', color=REF_COLOR, lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r'PED reference C$\beta$ ($\mathrm{\AA}$)')
    ax.set_ylabel(r'Pred C$\beta$ ($\mathrm{\AA}$)')
    setup_ax(ax)

    # Panel C: AA RMSD histogram
    ax = axes[1, 0]
    vals = df['aa_rmsd'].dropna().values
    ax.hist(vals, bins=50, color=C_PRED, alpha=0.7, density=True)
    ax.set_xlabel(r'All-atom RMSD ($\mathrm{\AA}$)'); ax.set_ylabel('Density')
    ax.text(0.95, 0.92, f'n={len(vals)}\nmed={np.median(vals):.2f}',
            transform=ax.transAxes, fontsize=8, va='top', ha='right')
    setup_ax(ax)

    # Panel D: BB RMSD histogram
    ax = axes[1, 1]
    vals = df['bb_rmsd'].dropna().values
    ax.hist(vals, bins=50, color=C_GT, alpha=0.7, density=True)
    ax.set_xlabel(r'Backbone RMSD ($\mathrm{\AA}$)'); ax.set_ylabel('Density')
    ax.text(0.95, 0.92, f'n={len(vals)}\nmed={np.median(vals):.2f}',
            transform=ax.transAxes, fontsize=8, va='top', ha='right')
    setup_ax(ax)

    # Panel E: SC RMSD histogram
    ax = axes[2, 0]
    vals = df['sc_rmsd'].dropna().values
    ax.hist(vals, bins=50, color=C_HIST, alpha=0.7, density=True)
    ax.set_xlabel(r'Side chain RMSD ($\mathrm{\AA}$)'); ax.set_ylabel('Density')
    ax.text(0.95, 0.92, f'n={len(vals)}\nmed={np.median(vals):.2f}',
            transform=ax.transAxes, fontsize=8, va='top', ha='right')
    setup_ax(ax)

    # Panel F: SC/BB ratio histogram
    ax = axes[2, 1]
    vals = df['sc_bb_ratio'].dropna().values
    vals = vals[vals < 20]
    ax.hist(vals, bins=50, color='#ff7f00', alpha=0.7, density=True)
    ax.set_xlabel('Side chain / Backbone ratio'); ax.set_ylabel('Density')
    ax.text(0.95, 0.92, f'med={np.median(vals):.1f}',
            transform=ax.transAxes, fontsize=8, va='top', ha='right')
    setup_ax(ax)

    # Panel G: Dmax pairwise scatter
    ax = axes[3, 0]
    gt = df['dmax_gt'].values
    pred = df['dmax_pred'].values
    mask = ~(np.isnan(gt) | np.isnan(pred))
    ax.scatter(gt[mask], pred[mask], s=1, alpha=0.15, c=C_PRED, rasterized=True)
    lim = [0, max(gt[mask].max(), pred[mask].max()) * 1.05]
    ax.plot(lim, lim, '--', color=REF_COLOR, lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r'PED reference $D_\mathrm{max}$ ($\mathrm{\AA}$)')
    ax.set_ylabel(r'Pred $D_\mathrm{max}$ ($\mathrm{\AA}$)')
    delta = np.abs(pred[mask] - gt[mask]).mean()
    rel = (np.abs(pred[mask] - gt[mask]) / gt[mask]).mean() * 100
    ax.text(0.05, 0.92, f'MAE={delta:.1f}\nRel={rel:.1f}%',
             transform=ax.transAxes, fontsize=8, va='top')
    setup_ax(ax)

    # Panel H: Helix fraction pairwise scatter
    ax = axes[3, 1]
    gt_h = df['dssp_helix_gt'] / df['dssp_total_gt'].replace(0, np.nan)
    pred_h = df['dssp_helix_pred'] / df['dssp_total_pred'].replace(0, np.nan)
    mask = ~(np.isnan(gt_h) | np.isnan(pred_h))
    ax.scatter(gt_h[mask], pred_h[mask], s=1, alpha=0.15, c=C_PRED, rasterized=True)
    ax.plot([0, 1], [0, 1], '--', color=REF_COLOR, lw=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('PED reference helix frac'); ax.set_ylabel('Pred helix frac')
    setup_ax(ax)

    for ax in axes.flat:
        ax.xaxis.label.set_fontsize(14)
        ax.yaxis.label.set_fontsize(14)
        ax.tick_params(axis='both', labelsize=11)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'figure_s1.png'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'figure_s1.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('  Supplementary figure saved (4x2 portrait, 8 panels A-H)')

def main():
    print('=' * 60)
    print('  Figure 2 Step=100 Plot')
    print('=' * 60)

    df, summary, sc_type, bond = load_data()
    print(f'  Data: {len(df)} conformers, {len(summary)} systems')

    plot_main_figure(df, summary, sc_type)
    plot_supplementary(df, summary, bond)

    print('\n  Plot complete.')

if __name__ == '__main__':
    main()
