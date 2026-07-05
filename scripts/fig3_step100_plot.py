"""Figure 3 Step=100: 5-panel main figure + 8-panel supplementary (4x2 portrait).

Main figure layout: 3 rows x 2 cols, Panel C spans full width.
  Row 1: A (Rg overlay) | B (CA fidelity)
  Row 2: C (JS divergence by residue class, full width)
  Row 3: D (Rotamer three-way) | E (Clash three-way)

Supplementary figure (3 rows x 2 cols):
  A: Bond geometry violins vs canonical reference values
  B: Cbeta deviation distribution
  C: Dmax distribution overlay (CG+CODLAD vs PED reference)
  D: Proline phi distribution
  E: Trans-proline favoured three-way comparison
  F: Helix fraction overlay (CG+CODLAD vs PED reference)

Style: Arial 16/18 pt, 600 DPI, colorblind-safe palette, top/right spines off.
Panel labels omitted (added in manuscript layout).
"""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

PROJECT = Path(__file__).resolve().parents[1]
OUT = os.path.join(PROJECT, 'logs/figure3_step100')
FIG2_OUT = os.path.join(PROJECT, 'logs/figure2_step100')
FIGDIR = os.path.join(PROJECT, 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# Colorblind-safe palette (Wong, 2011)
C_PED_REF    = '#0072B2'   # Blue — PED reference / ground truth
C_PED_CODLAD = '#009E73'   # Green — PED+CODLAD
C_CG_CODLAD  = '#D55E00'   # Vermilion — CG+CODLAD
C_NEUTRAL    = '#999999'   # Grey — pipeline / reference lines

COLORS = {
    'ped_ref':    C_PED_REF,
    'ped_codlad': C_PED_CODLAD,
    'cg_codlad':  C_CG_CODLAD,
}

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
    'legend.frameon':     False,
})


def load_data():
    cg = pd.read_csv(os.path.join(OUT, 'per_frame_metrics.csv'))
    fig2 = pd.read_csv(os.path.join(FIG2_OUT, 'per_conformer_metrics.csv'))
    fig2_sys = pd.read_csv(os.path.join(FIG2_OUT, 'per_system_summary.csv'))
    cg_sys = pd.read_csv(os.path.join(OUT, 'per_system_summary.csv'))
    return cg, fig2, fig2_sys, cg_sys


# ═══════════════════════════════════════════════════════════════════════════════════
#  Main figure panels
# ═══════════════════════════════════════════════════════════════════════════════════

def panel_a_rg_overlay(ax, cg, fig2):
    cg_rg = cg['rg'].dropna().values
    ref_rg = fig2['rg_gt'].dropna().values
    lo = min(cg_rg.min(), ref_rg.min())
    hi = max(cg_rg.max(), ref_rg.max())
    bins = np.linspace(lo, hi, 50)
    ax.hist(ref_rg, bins=bins, alpha=0.5, color=COLORS['ped_ref'],
            label='PED reference', density=True)
    ax.hist(cg_rg, bins=bins, alpha=0.5, color=COLORS['cg_codlad'],
            label='CG+CODLAD', density=True)
    ax.set_xlabel(r'$R_\mathrm{g}$ ($\mathrm{\AA}$)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=12, loc='upper right')
    ax.tick_params(labelsize=14)


def panel_b_ca_fidelity(ax, cg):
    vals = cg['ca_mean_deviation'].dropna().values
    if len(vals) == 0:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
        return
    ax.hist(vals, bins=50, color=COLORS['cg_codlad'], alpha=0.7)
    ax.set_xlabel(r'C$\alpha$ deviation ($\mathrm{\AA}$)')
    ax.set_ylabel('Count')
    ax.tick_params(labelsize=14)
    ax.text(0.97, 0.95, f'mean = {vals.mean():.4f} $\mathrm{{\AA}}$',
            transform=ax.transAxes, fontsize=11, ha='right', va='top')


def panel_c_js_divergence(ax):
    div_path = os.path.join(OUT, 'rama_divergence_three_condition.csv')
    if not os.path.exists(div_path):
        ax.text(0.5, 0.5, 'No divergence data', transform=ax.transAxes, ha='center')
        return

    df = pd.read_csv(div_path)
    class_order = ['Pooled', 'General', 'Glycine', 'Pre-Pro', 'Ile/Val', 'Trans-Pro', 'Cis-Pro']
    df_ordered = df.set_index('residue_class').loc[class_order].reset_index()

    x = np.arange(len(class_order))
    w = 0.25

    pipeline = df_ordered['JS_pipeline'].values
    total    = df_ordered['JS_total'].values
    cg_attr  = df_ordered['JS_cg'].values

    ax.bar(x - w, pipeline, w, color=C_NEUTRAL, alpha=0.85,
           label='JS(PED+CODLAD, PED ref.)')
    ax.bar(x,      total,    w, color=C_PED_REF, alpha=0.85,
           label='JS(CG+CODLAD, PED ref.)')
    ax.bar(x + w,  cg_attr,  w, color=C_CG_CODLAD, alpha=0.85,
           label=r'$\Delta JS_\mathrm{CG}$ = total - pipeline')

    ax.set_xticks(x)
    ax.set_xticklabels(class_order, rotation=30, ha='right', fontsize=12)
    ax.set_ylabel('JS divergence / increment')
    ax.set_ylim(0, 0.52)
    ax.legend(fontsize=11, loc='upper left')
    ax.tick_params(labelsize=14)


def panel_d_rotamer(ax, cg_sys, fig2_sys):
    cg_val    = cg_sys['rota_favored_pct'].mean()
    fig2_pred = (fig2_sys['rota_favored_pct_pred'].mean()
                 if 'rota_favored_pct_pred' in fig2_sys.columns else 0)
    fig2_gt   = (fig2_sys['rota_favored_pct_gt'].mean()
                 if 'rota_favored_pct_gt' in fig2_sys.columns else 0)

    w = 0.25
    ax.bar([-w], [fig2_gt],  w, color=COLORS['ped_ref'],    alpha=0.9, label='PED reference')
    ax.bar([0],  [fig2_pred], w, color=COLORS['ped_codlad'], alpha=0.9, label='PED+CODLAD')
    ax.bar([w],  [cg_val],    w, color=COLORS['cg_codlad'],  alpha=0.9, label='CG+CODLAD')
    ax.set_ylabel('Rotamer favoured (%)')
    ax.set_xticks([])
    ax.legend(fontsize=12, loc='upper right', bbox_to_anchor=(1.02, 1.08))
    ax.tick_params(labelsize=14)
    ax.set_ylim(0, 110)


def panel_e_clash(ax, cg_sys, fig2_sys):
    systems = sorted(cg_sys['ped_id'].unique())
    cg_vals   = [cg_sys[cg_sys['ped_id'] == s]['clash_mean'].values[0] for s in systems]
    fig2_pred = [fig2_sys[fig2_sys['ped_id'] == s]['clash_pred_mean'].values[0]
                 if len(fig2_sys[fig2_sys['ped_id'] == s]) > 0 else 0 for s in systems]
    fig2_gt   = [fig2_sys[fig2_sys['ped_id'] == s]['clash_gt_mean'].values[0]
                 if len(fig2_sys[fig2_sys['ped_id'] == s]) > 0 else 0 for s in systems]

    x = np.arange(len(systems))
    w = 0.25
    ax.bar(x - w, fig2_gt,  w, color=COLORS['ped_ref'],    alpha=0.9, label='PED reference')
    ax.bar(x,     fig2_pred, w, color=COLORS['ped_codlad'], alpha=0.9, label='PED+CODLAD')
    ax.bar(x + w,  cg_vals,   w, color=COLORS['cg_codlad'],  alpha=0.9, label='CG+CODLAD')
    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Clash count')
    ax.legend(fontsize=12, loc='upper left')
    ax.tick_params(labelsize=14)


# ═══════════════════════════════════════════════════════════════════════════════════
#  Main Figure: 2 rows × 2 cols (4 panels)
#    A: Rg overlay        |  B: JS divergence
#    C: Rotamer three-way  |  D: Clash three-way
# ═══════════════════════════════════════════════════════════════════════════════════

def main_figure():
    cg, fig2, fig2_sys, cg_sys = load_data()

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3,
                  width_ratios=[0.4, 0.6])

    # Row 1: A (Rg overlay) | B (JS divergence)
    panel_a_rg_overlay(fig.add_subplot(gs[0, 0]), cg, fig2)
    panel_c_js_divergence(fig.add_subplot(gs[0, 1]))

    # Row 2: C (Rotamer) | D (Clash)
    panel_d_rotamer(fig.add_subplot(gs[1, 0]), cg_sys, fig2_sys)
    panel_e_clash(fig.add_subplot(gs[1, 1]), cg_sys, fig2_sys)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'figure3.png'), dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'figure3.pdf'), dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(FIGDIR, 'figure3.png'), dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(FIGDIR, 'figure3.pdf'), dpi=600, bbox_inches='tight')
    plt.close(fig)
    print('  Main figure saved (2x2, 4 panels A-D)')


# ═══════════════════════════════════════════════════════════════════════════════════
#  Supplementary Figure: 4 rows × 2 cols portrait
# ═══════════════════════════════════════════════════════════════════════════════════

def supp_figure():
    cg, fig2, fig2_sys, cg_sys = load_data()

    # 3 rows × 2 cols, 6 panels (A–F)
    fig, axes = plt.subplots(3, 2, figsize=(11, 14))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    # ── A: Bond geometry violins ────────────────────────────────────────────
    ax = axes[0, 0]
    canonical = {'CA_C': 1.53, 'N_CA': 1.46, 'C_N': 1.33}
    bond_labels = {'CA_C': 'CA$-$C', 'N_CA': 'N$-$CA', 'C_N': 'C$-$N'}
    data = []; labels = []
    for bt in ['CA_C', 'N_CA', 'C_N']:
        col = f'bond_{bt}'
        if col in cg.columns:
            vals = cg[col].dropna().values
            data.append(vals)
            labels.append(bond_labels[bt])
    if data:
        parts = ax.violinplot(data, positions=range(len(data)), showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor(COLORS['cg_codlad']); pc.set_alpha(0.6)
        for i, bt in enumerate(['CA_C', 'N_CA', 'C_N']):
            ax.axhline(canonical[bt], color=C_NEUTRAL, ls='--', lw=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(r'Bond length ($\mathrm{\AA}$)')
    ax.tick_params(labelsize=11)

    # ── B: Cβ deviation distribution ────────────────────────────────────────
    ax = axes[0, 1]
    cbdev = cg['cbdev_mean'].dropna().values
    if len(cbdev):
        ax.hist(cbdev, bins=40, color=COLORS['cg_codlad'], alpha=0.7)
    ax.set_xlabel(r'C$\beta$ deviation ($\mathrm{\AA}$)')
    ax.set_ylabel('Count')
    ax.tick_params(labelsize=11)

    # ── C: Dmax distribution overlay ────────────────────────────────────────
    ax = axes[1, 0]
    cg_dmax = cg['dmax'].dropna().values
    ref_dmax = fig2['dmax_gt'].dropna().values if 'dmax_gt' in fig2.columns else np.array([])
    if len(ref_dmax) and len(cg_dmax):
        lo = min(cg_dmax.min(), ref_dmax.min())
        hi = max(cg_dmax.max(), ref_dmax.max())
        bins = np.linspace(lo, hi, 50)
        ax.hist(ref_dmax, bins=bins, alpha=0.5, color=COLORS['ped_ref'],
                label='PED reference', density=True)
        ax.hist(cg_dmax, bins=bins, alpha=0.5, color=COLORS['cg_codlad'],
                label='CG+CODLAD', density=True)
    elif len(cg_dmax):
        ax.hist(cg_dmax, bins=50, alpha=0.5, color=COLORS['cg_codlad'],
                label='CG+CODLAD', density=True)
    ax.set_xlabel(r'$D_\mathrm{max}$ ($\mathrm{\AA}$)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=9, loc='upper right')
    ax.tick_params(labelsize=11)

    # ── D: Proline φ distribution ───────────────────────────────────────────
    ax = axes[1, 1]
    pro_phi = cg['pro_phi_mean'].dropna().values
    if len(pro_phi):
        ax.hist(pro_phi, bins=40, color=COLORS['cg_codlad'], alpha=0.7, density=True)
    ax.set_xlabel(r'Proline $\varphi$ (°)')
    ax.set_ylabel('Density')
    ax.tick_params(labelsize=11)

    # ── E: Trans-proline favoured three-way ─────────────────────────────────
    ax = axes[2, 0]
    rama3_path = os.path.join(OUT, 'rama_three_way.csv')
    if os.path.exists(rama3_path):
        rama3 = pd.read_csv(rama3_path)
        tp = rama3[rama3['category'] == 'trans_proline']
        if len(tp):
            vals   = [tp['PED_ref'].values[0], tp['PED_CODLAD'].values[0],
                      tp['CG_CODLAD'].values[0]]
            labels = ['PED ref', 'PED+CODLAD', 'CG+CODLAD']
            clrs   = [COLORS['ped_ref'], COLORS['ped_codlad'], COLORS['cg_codlad']]
            ax.bar(range(3), vals, color=clrs, alpha=0.9, edgecolor='white', lw=0.3)
            ax.set_xticks(range(3))
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylabel('Trans-Pro favoured (%)')
            ax.set_ylim(0, 110)
    ax.tick_params(labelsize=11)

    # ── F: Helix fraction distribution overlay ──────────────────────────────
    ax = axes[2, 1]
    cg_h = (cg['dssp_helix'] / cg['dssp_total']).replace([np.inf, -np.inf], np.nan).dropna().values
    ref_h_raw = (fig2['dssp_helix_gt'].values / fig2['dssp_total_gt'].values
                 if 'dssp_helix_gt' in fig2.columns else np.array([]))
    ref_h = ref_h_raw[~np.isnan(ref_h_raw)]
    if len(ref_h):
        bins = np.linspace(0, 0.2, 30)
        ax.hist(ref_h, bins=bins, alpha=0.5, color=COLORS['ped_ref'],
                label='PED reference', density=True)
    if len(cg_h):
        ax.hist(cg_h, bins=30, alpha=0.5, color=COLORS['cg_codlad'],
                label='CG+CODLAD', density=True)
    ax.set_xlim(0, 0.2)
    ax.set_xlabel('Helix fraction')
    ax.set_ylabel('Density')
    ax.legend(fontsize=9, loc='upper right')
    ax.tick_params(labelsize=11)

    # Override supplementary label/tick sizes for manuscript readability
    for ax in axes.flat:
        if ax.get_visible():
            ax.xaxis.label.set_fontsize(14)
            ax.yaxis.label.set_fontsize(14)
            ax.tick_params(axis='both', labelsize=11)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'figure_s3.png'), dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'figure_s3.pdf'), dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(FIGDIR, 'figure_s3.png'), dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(FIGDIR, 'figure_s3.pdf'), dpi=600, bbox_inches='tight')
    plt.close(fig)
    print('  Supplementary figure saved (3x2, 6 panels A-F)')


if __name__ == '__main__':
    main_figure()
    supp_figure()
