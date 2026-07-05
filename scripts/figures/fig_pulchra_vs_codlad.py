#!/usr/bin/env python3
"""Plot PULCHRA versus CODLAD comparison for Reviewer 1 rebuttal.

Reads:
  logs/figure2_step100/per_system_summary.csv      PED+CODLAD
  logs/figure3_step100/per_system_summary.csv      CG+CODLAD, clash fixed by KDTree recompute
  logs/pulchra_ped/per_system_summary.csv          PED+PULCHRA
  logs/pulchra_cg/per_system_summary.csv           CG+PULCHRA

Writes:
  manuscript/figures/pulchra_vs_codlad_rebuttal.{png,svg}
  manuscript/tables/pulchra_vs_codlad_summary.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[2]


def mean_col(df, col):
    return float(pd.to_numeric(df[col], errors='coerce').mean())


def sem_col(df, col):
    x = pd.to_numeric(df[col], errors='coerce').dropna().to_numpy(float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def load_data():
    codlad_ped = pd.read_csv(PROJECT / 'logs/figure2_step100/per_system_summary.csv')
    codlad_cg = pd.read_csv(PROJECT / 'logs/figure3_step100/per_system_summary.csv')
    pul_ped = pd.read_csv(PROJECT / 'logs/pulchra_ped/per_system_summary.csv')
    pul_cg = pd.read_csv(PROJECT / 'logs/pulchra_cg/per_system_summary.csv')
    main23 = sorted(set(pul_ped.ped_id) & set(pul_cg.ped_id) & set(codlad_ped.ped_id) & set(codlad_cg.ped_id))
    return (
        codlad_ped[codlad_ped.ped_id.isin(main23)].copy(),
        codlad_cg[codlad_cg.ped_id.isin(main23)].copy(),
        pul_ped[pul_ped.ped_id.isin(main23)].copy(),
        pul_cg[pul_cg.ped_id.isin(main23)].copy(),
        main23,
    )


def build_summary():
    codlad_ped, codlad_cg, pul_ped, pul_cg, main23 = load_data()
    rows = []
    specs = [
        ('PED+CODLAD', codlad_ped, {
            'Rg (A)': 'rg_pred_mean',
            'Rama favored (%)': 'rama_favored_pct_pred',
            'Rotamer favored (%)': 'rota_favored_pct_pred',
            'Cbeta deviation (A)': 'cbdev_mean_pred_mean',
            'Helix fraction (%)': 'helix_frac_pred_pct',
            'Trans-Pro phi (deg)': 'pro_phi_mean_pred',
            'Clash count': 'clash_pred_mean',
        }),
        ('CG+CODLAD', codlad_cg, {
            'Rg (A)': 'rg_mean',
            'Rama favored (%)': 'rama_favored_pct',
            'Rotamer favored (%)': 'rota_favored_pct',
            'Cbeta deviation (A)': 'cbdev_mean',
            'Helix fraction (%)': 'helix_frac_pct',
            'Trans-Pro phi (deg)': 'pro_phi_mean',
            'Clash count': 'clash_mean',
        }),
        ('PED+PULCHRA', pul_ped, {
            'Rg (A)': 'rg_mean',
            'Rama favored (%)': 'rama_favored_pct',
            'Rotamer favored (%)': 'rota_favored_pct',
            'Cbeta deviation (A)': 'cbdev_mean',
            'Helix fraction (%)': 'helix_frac_pct',
            'Trans-Pro phi (deg)': 'pro_phi_mean',
            'Clash count': 'clash_mean',
        }),
        ('CG+PULCHRA', pul_cg, {
            'Rg (A)': 'rg_mean',
            'Rama favored (%)': 'rama_favored_pct',
            'Rotamer favored (%)': 'rota_favored_pct',
            'Cbeta deviation (A)': 'cbdev_mean',
            'Helix fraction (%)': 'helix_frac_pct',
            'Trans-Pro phi (deg)': 'pro_phi_mean',
            'Clash count': 'clash_mean',
        }),
    ]

    # Convert fractions to percent in copied columns to keep mapping simple.
    codlad_ped['helix_frac_pred_pct'] = codlad_ped['helix_frac_pred'] * 100.0
    codlad_cg['helix_frac_pct'] = codlad_cg['helix_frac'] * 100.0
    pul_ped['helix_frac_pct'] = pul_ped['helix_frac'] * 100.0
    pul_cg['helix_frac_pct'] = pul_cg['helix_frac'] * 100.0

    # Rebuild specs after adding derived columns.
    specs[0] = (specs[0][0], codlad_ped, specs[0][2])
    specs[1] = (specs[1][0], codlad_cg, specs[1][2])
    specs[2] = (specs[2][0], pul_ped, specs[2][2])
    specs[3] = (specs[3][0], pul_cg, specs[3][2])

    for condition, df, mapping in specs:
        method = 'CODLAD' if 'CODLAD' in condition else 'PULCHRA'
        input_type = 'PED input' if condition.startswith('PED') else 'CALVADOS input'
        for metric, col in mapping.items():
            rows.append({
                'condition': condition,
                'method': method,
                'input': input_type,
                'metric': metric,
                'mean_per_system': mean_col(df, col),
                'sem_per_system': sem_col(df, col),
                'n_systems': len(df),
            })
    summary = pd.DataFrame(rows)
    return summary, main23


def get(summary, condition, metric):
    row = summary[(summary.condition == condition) & (summary.metric == metric)].iloc[0]
    return row.mean_per_system, row.sem_per_system


def plot(summary, main23):
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 9,
        'axes.linewidth': 0.9,
        'axes.edgecolor': '0.2',
    })
    metrics = [
        ('Rotamer favored (%)', 'Rotamer favored (%)', (75, 100), 'Higher is better'),
        ('Cbeta deviation (A)', r'C$\beta$ deviation (A)', (0, 0.85), 'Lower is better'),
        ('Clash count', 'Clash count per conformer', (0, 450), 'Lower is better'),
        ('Helix fraction (%)', 'Helix fraction (%)', (0, 4.0), 'Context dependent'),
        ('Rama favored (%)', 'Ramachandran favored (%)', (82, 94), 'Higher is better'),
        ('Trans-Pro phi (deg)', r'Trans-Pro $\phi$ mean (deg)', (-95, -25), 'PED reference approx. -67 deg'),
    ]
    conditions = ['PED+CODLAD', 'PED+PULCHRA', 'CG+CODLAD', 'CG+PULCHRA']
    colors = {'CODLAD': '#1f77b4', 'PULCHRA': '#ff7f0e'}
    hatches = {'PED+CODLAD': '', 'PED+PULCHRA': '', 'CG+CODLAD': '//', 'CG+PULCHRA': '//'}

    fig, axs = plt.subplots(2, 3, figsize=(10.2, 6.1))
    for ax, (metric, ylabel, ylim, note) in zip(axs.ravel(), metrics):
        x = np.arange(len(conditions))
        vals, errs, bar_colors = [], [], []
        for cond in conditions:
            v, e = get(summary, cond, metric)
            vals.append(v); errs.append(e)
            bar_colors.append(colors['CODLAD' if 'CODLAD' in cond else 'PULCHRA'])
        if metric == 'Trans-Pro phi (deg)':
            for xi, v, e, c, cond in zip(x, vals, errs, bar_colors, conditions):
                marker = 'o' if 'PED' in cond else 's'
                ax.errorbar([xi], [v], yerr=[e], marker=marker, ms=6.5, capsize=3,
                            color=c, markerfacecolor=c, markeredgecolor='0.2', lw=1.2)
            ax.axhline(-67.4, color='0.35', lw=1.0, ls='--')
            ax.text(0.98, 0.88, 'PED reference approx. -67 deg', transform=ax.transAxes,
                    ha='right', va='top', fontsize=7.2, color='0.35',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.72, pad=1.2))
        else:
            bars = ax.bar(x, vals, yerr=errs, capsize=2.5, color=bar_colors,
                          edgecolor='0.2', linewidth=0.6)
            for b, cond in zip(bars, conditions):
                b.set_hatch(hatches[cond])
            ax.text(0.98, 0.96, note, transform=ax.transAxes, ha='right', va='top',
                    fontsize=7.2, color='0.35',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.72, pad=1.2))
        ax.set_xticks(x)
        ax.set_xticklabels(['PED\nCODLAD', 'PED\nPULCHRA', 'CG\nCODLAD', 'CG\nPULCHRA'], fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.set_title(metric.replace('Cbeta', r'C$\beta$'), loc='left', fontweight='bold', fontsize=9.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=8)

    fig.suptitle(f'Classical PULCHRA versus learned CODLAD backmapping (n={len(main23)} systems)',
                 fontsize=12.5, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out_base = PROJECT / 'manuscript/figures/figure_s6_pulchra_vs_codlad'
    for ext in ['png', 'svg', 'pdf']:
        out = out_base.with_suffix(f'.{ext}')
        fig.savefig(out, dpi=400, bbox_inches='tight', facecolor='white')
        print(f'saved {out}')


def main():
    summary, main23 = build_summary()
    out_csv = PROJECT / 'manuscript/tables/pulchra_vs_codlad_summary.csv'
    summary.to_csv(out_csv, index=False)
    print(f'saved {out_csv}')
    print(summary.pivot(index='metric', columns='condition', values='mean_per_system').round(3).to_string())
    plot(summary, main23)


if __name__ == '__main__':
    main()
