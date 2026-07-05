"""Figure 4 Step=100: Save each panel as a separate figure (no labels).

Panels: 5 per-system helix profiles (A1-A5), p15PAF zoom (B),
        Tif2 NRID zoom (S1), ACTR zoom (S2), step comparison (S3).
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT = '/MDdata/data04/jxhuang/cg_cascade'
OUT = os.path.join(PROJECT, 'logs/figure4_step100')
PANEL_DIR = os.path.join(OUT, 'panels')
os.makedirs(PANEL_DIR, exist_ok=True)

ACCENT = '#2166ac'
GREY = '#999999'
PRED_COLOR = '#333333'

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 18, "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14,
    "figure.dpi": 600, "savefig.dpi": 600, "mathtext.default": "regular",
})


def save_panel(fig, name):
    fig.savefig(os.path.join(PANEL_DIR, f'figure4_panel_{name}.png'),
                dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(PANEL_DIR, f'figure4_panel_{name}.pdf'),
                bbox_inches='tight')
    plt.close(fig)
    print(f'  {name} saved')


def load_data():
    df_all = pd.read_csv(os.path.join(OUT, 'per_residue_helix.csv'))
    df_summary = pd.read_csv(os.path.join(OUT, 'per_system_summary.csv'))

    df_summary = df_summary.sort_values('ref_mean_helix_pct', ascending=False)

    p15paf_z = df_all[(df_all['system'] == 'PED00016') &
                       (df_all['residue_index'] >= 54) & (df_all['residue_index'] <= 60)]
    actr_z = df_all[(df_all['system'] == 'PED00536') & (df_all['residue_index'] <= 30)]
    ped184_z = df_all[(df_all['system'] == 'PED00184') &
                       (df_all['residue_index'] >= 120) & (df_all['residue_index'] <= 140)]

    return df_all, df_summary, p15paf_z, actr_z, ped184_z


# ── Panel A1-A5: Per-system helix profiles (5 separate panels) ────────────────
def panel_per_system(df_all, df_summary, ref_name, panel_name):
    fig, ax = plt.subplots(figsize=(16, 4))

    sub = df_all[df_all['system'] == ref_name]
    x = sub['residue_index'].values
    pred = sub['pred_helix'].values * 100
    ref_val = sub['ref_helix'].values * 100

    has_ref = not np.all(np.isnan(ref_val))
    if has_ref:
        ax.fill_between(x, 0, ref_val, color=ACCENT, alpha=0.3, label='Reference')
        ax.plot(x, ref_val, '-', color=ACCENT, linewidth=1.2)
        ymax = max(np.nanmax(ref_val) * 1.4, 15.0)
    else:
        ymax = 15.0
    ax.plot(x, pred, '-', color=PRED_COLOR, linewidth=1.0, label='Predicted (step=100)')

    if ref_name == 'PED00016':
        ax.axvspan(54, 60, color=GREY, alpha=0.12)
    if ref_name == 'PED00536':
        ax.axvspan(1, 30, color=GREY, alpha=0.12)
    if ref_name == 'PED00184':
        ax.axvspan(120, 140, color=GREY, alpha=0.12)

    sr = df_summary[df_summary['system'] == ref_name]
    if len(sr) > 0:
        r_val = sr.iloc[0]['pearson_r']
        label_str = sr.iloc[0]['label']
        label = f'{ref_name} ({label_str})'
        if not np.isnan(r_val):
            label += f', r={r_val:.2f}'
        ax.text(0.98, 0.85, label, transform=ax.transAxes, ha='right', fontsize=12)

    ax.set_ylim(-0.5, ymax)
    ax.set_ylabel('Helix (%)')
    ax.set_xlabel('Residue')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', fontsize=10)
    save_panel(fig, panel_name)


# ── Panel B: p15PAF residues 54-60 zoom ───────────────────────────────────────
def panel_b(p15paf_z):
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(p15paf_z) > 0:
        x_pos = p15paf_z['residue_index'].values
        ref_vals = p15paf_z['ref_helix'].values * 100
        pred_vals = p15paf_z['pred_helix'].values * 100
        width = 0.35
        ax.bar(x_pos - width/2, ref_vals, width, color=ACCENT, label='Reference')
        ax.bar(x_pos + width/2, pred_vals, width, color=PRED_COLOR, label='Predicted')
        if 59 in x_pos:
            idx59 = np.where(x_pos == 59)[0][0]
            ax.annotate('TRP 59\n17.6% ref', xy=(59, ref_vals[idx59]),
                        xytext=(59.5, ref_vals[idx59] + 3),
                        fontsize=10, ha='left',
                        arrowprops=dict(arrowstyle='->', color='black'))
    ax.set_xlabel('Residue')
    ax.set_ylabel('Helix fraction (%)')
    ax.set_ylim(0, 22)
    ax.set_xticks(range(54, 61))
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    save_panel(fig, 'B_p15PAF_zoom')


# ── S1: PED00184 residue 120-140 zoom ─────────────────────────────────────────
def panel_s1(ped184_z):
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(ped184_z) > 0:
        x_pos = ped184_z['residue_index'].values
        ref_vals = ped184_z['ref_helix'].values * 100
        pred_vals = ped184_z['pred_helix'].values * 100
        width = 0.35
        ax.bar(x_pos - width/2, ref_vals, width, color=ACCENT, label='Reference')
        ax.bar(x_pos + width/2, pred_vals, width, color=PRED_COLOR, label='Predicted')
    ax.set_xlabel('Residue')
    ax.set_ylabel('Helix (%)')
    ax.set_title('PED00184 (Tif2 NRID) residue 120-140')
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    save_panel(fig, 'S1_Tif2NRID_zoom')


# ── S2: ACTR N-terminal helix zoom ────────────────────────────────────────────
def panel_s2(actr_z):
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(actr_z) > 0:
        x_pos = actr_z['residue_index'].values
        ref_vals = actr_z['ref_helix'].values * 100
        pred_vals = actr_z['pred_helix'].values * 100
        width = 0.35
        ax.bar(x_pos - width/2, ref_vals, width, color=ACCENT, label='Reference')
        ax.bar(x_pos + width/2, pred_vals, width, color=PRED_COLOR, label='Predicted')
    ax.set_xlabel('Residue')
    ax.set_ylabel('Helix (%)')
    ax.set_title('ACTR (PED00536) N-terminal helix')
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    save_panel(fig, 'S2_ACTR_zoom')


# ── S3: Step=5 vs step=100 comparison ─────────────────────────────────────────
def panel_s3(df_summary):
    fig, ax = plt.subplots(figsize=(8, 5))
    systems = df_summary['system'].values
    step5_vals = df_summary['step5_pred_mean_helix_pct'].values
    step100_vals = df_summary['pred_mean_helix_pct'].values
    x_pos = np.arange(len(systems))
    width = 0.35
    ax.bar(x_pos - width/2, step5_vals, width, color='#7570b3', label='Step=5')
    ax.bar(x_pos + width/2, step100_vals, width, color='#d95f02', label='Step=100')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(systems, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Mean predicted helix (%)')
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    save_panel(fig, 'S3_step_comparison')


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    df_all, df_summary, p15paf_z, actr_z, ped184_z = load_data()

    print('Per-system helix profiles (sorted by ref helix, descending):')
    panel_labels = {
        'PED00184': 'A1_PED00184_Tif2NRID',
        'PED00536': 'A2_PED00536_ACTR',
        'PED00016': 'A3_PED00016_p15PAF',
        'PED00229': 'A4_PED00229_p53TAD1',
        'PED00003': 'A5_PED00003_BetaSynuclein',
    }
    for ref_name in df_summary['system'].values:
        label = panel_labels.get(ref_name, ref_name)
        panel_per_system(df_all, df_summary, ref_name, label)

    print('\nZoom and comparison panels:')
    panel_b(p15paf_z)
    panel_s1(ped184_z)
    panel_s2(actr_z)
    panel_s3(df_summary)

    print(f'\nAll panels saved to {PANEL_DIR}/')
