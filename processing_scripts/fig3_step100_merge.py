"""Merge Figure 3 step=100 results, load Figure 2 step=100 for three-condition decomposition.

Three conditions:
  1. PED reference (from Figure 2 GT columns)
  2. PED + CODLAD (from logs/figure2_step100/)
  3. CG + CODLAD (from logs/figure3_step100/)

Usage: python scripts/fig3_step100_merge.py
"""
import os, sys, csv, json, glob
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

PROJECT = '/MDdata/data04/jxhuang/cg_cascade'
OUT = os.path.join(PROJECT, 'logs/figure3_step100')
FIG2_OUT = os.path.join(PROJECT, 'logs/figure2_step100')

with open(os.path.join(PROJECT, 'dataset/clean_v5.csv')) as f:
    CLEAN_V5 = sorted({r['ped_id'] for r in csv.DictReader(f)})

RAMA_CATS = ['general', 'glycine', 'pre_proline', 'trans_proline', 'cis_proline', 'ile_val']

def load_cg_results():
    """Load CG+CODLAD results from per-system CSVs or GPU JSON files."""
    # Try per-system CSVs first (more reliable, written by worker)
    csv_files = sorted(glob.glob(os.path.join(OUT, '*_per_frame.csv')))
    if csv_files:
        dfs = [pd.read_csv(f) for f in csv_files]
        df = pd.concat(dfs, ignore_index=True)
        df = df[df['ped_id'].isin(CLEAN_V5)]
        print(f'  Loaded {len(df)} rows from {len(csv_files)} per-system CSVs')
        return df

    # Fallback: GPU JSON files
    all_rows = []
    for gpu_id in range(4):
        path = os.path.join(OUT, f'results_gpu{gpu_id}.json')
        if not os.path.exists(path): continue
        with open(path) as f:
            rows = json.load(f)
        for r in rows:
            if r.get('ped_id') in CLEAN_V5:
                all_rows.append(r)
    if all_rows:
        df = pd.DataFrame(all_rows)
        print(f'  Loaded {len(df)} rows from GPU JSON files')
        return df

    # Fallback: read from per-frame CSV if it exists
    pf_path = os.path.join(OUT, 'per_frame_metrics.csv')
    if os.path.exists(pf_path):
        df = pd.read_csv(pf_path)
        print(f'  Loaded {len(df)} rows from per_frame_metrics.csv')
        return df

    return pd.DataFrame()

def load_fig2_per_conformer():
    """Load Figure 2 step=100 per-conformer metrics."""
    path = os.path.join(FIG2_OUT, 'per_conformer_metrics.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

def load_fig2_summary():
    """Load Figure 2 step=100 per-system summary."""
    path = os.path.join(FIG2_OUT, 'per_system_summary.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

def build_per_system_summary(df):
    summary_rows = []
    for ped_id in sorted(df['ped_id'].unique()):
        sub = df[df['ped_id'] == ped_id]
        row = {'ped_id': ped_id, 'n_frames': len(sub)}

        # Rg
        vals = sub['rg'].dropna()
        row['rg_mean'] = vals.mean() if len(vals) else np.nan
        row['rg_std'] = vals.std() if len(vals) else np.nan

        # Dmax
        vals = sub['dmax'].dropna()
        row['dmax_mean'] = vals.mean() if len(vals) else np.nan

        # Ramachandran pooled
        fav = sub['rama_pooled_fav'].sum()
        tot = sub['rama_pooled_tot'].sum()
        row['rama_favored_pct'] = fav / tot * 100 if tot else np.nan

        # Rotamer
        fav = sub['rota_fav'].sum()
        tot = sub['rota_tot'].sum()
        row['rota_favored_pct'] = fav / tot * 100 if tot else np.nan

        # Cbdev
        vals = sub['cbdev_mean'].dropna()
        row['cbdev_mean'] = vals.mean() if len(vals) else np.nan

        # Clash
        vals = sub['clash'].dropna()
        row['clash_mean'] = vals.mean() if len(vals) else np.nan

        # Helix
        h = sub['dssp_helix'].sum()
        t = sub['dssp_total'].sum()
        row['helix_frac'] = h / t if t else np.nan

        # CA fidelity
        vals = sub['ca_mean_deviation'].dropna()
        row['ca_mean_dev'] = vals.mean() if len(vals) else np.nan

        # Pro phi
        vals = sub['pro_phi_mean'].dropna()
        row['pro_phi_mean'] = vals.mean() if len(vals) else np.nan

        # Bond geometry
        for bt in ['CA_C', 'N_CA', 'C_N']:
            col = f'bond_{bt}'
            vals = sub[col].dropna() if col in sub.columns else pd.Series()
            row[f'bond_{bt}_mean'] = vals.mean() if len(vals) else np.nan

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    path = os.path.join(OUT, 'per_system_summary.csv')
    summary.to_csv(path, index=False)
    print(f'  Per-system: {len(summary)} systems -> {path}')
    return summary

def build_rama_by_class(df):
    rows = []
    for ped_id in sorted(df['ped_id'].unique()):
        sub = df[df['ped_id'] == ped_id]
        for cat in RAMA_CATS:
            fav_col = f'rama_{cat}_fav'
            tot_col = f'rama_{cat}_tot'
            if fav_col not in sub.columns: continue
            fav = sub[fav_col].sum()
            tot = sub[tot_col].sum()
            rows.append({
                'ped_id': ped_id, 'category': cat,
                'favored': fav, 'total': tot,
                'favored_pct': fav / tot * 100 if tot else np.nan,
            })
    rama_df = pd.DataFrame(rows)
    path = os.path.join(OUT, 'rama_by_class.csv')
    rama_df.to_csv(path, index=False)
    print(f'  Rama by class: {len(rama_df)} rows -> {path}')
    return rama_df

def build_bond_geometry(df):
    rows = []
    for ped_id in sorted(df['ped_id'].unique()):
        sub = df[df['ped_id'] == ped_id]
        for bt in ['CA_C', 'N_CA', 'C_N']:
            col = f'bond_{bt}'
            vals = sub[col].dropna() if col in sub.columns else pd.Series()
            rows.append({
                'ped_id': ped_id, 'bond_type': bt,
                'mean': vals.mean() if len(vals) else np.nan,
                'std': vals.std() if len(vals) else np.nan,
            })
    bond_df = pd.DataFrame(rows)
    path = os.path.join(OUT, 'bond_geometry.csv')
    bond_df.to_csv(path, index=False)
    return bond_df

def build_ca_fidelity(df):
    rows = []
    for ped_id in sorted(df['ped_id'].unique()):
        sub = df[df['ped_id'] == ped_id]
        vals = sub['ca_mean_deviation'].dropna()
        rows.append({
            'ped_id': ped_id,
            'mean_deviation': vals.mean() if len(vals) else np.nan,
            'max_deviation': sub['ca_max_deviation'].max() if 'ca_max_deviation' in sub.columns else np.nan,
        })
    ca_df = pd.DataFrame(rows)
    path = os.path.join(OUT, 'ca_fidelity.csv')
    ca_df.to_csv(path, index=False)
    return ca_df

def build_rg_ks_tests(df, fig2_conf):
    """KS test between CG+CODLAD Rg and PED reference Rg per system."""
    rows = []
    for ped_id in sorted(df['ped_id'].unique()):
        sub = df[df['ped_id'] == ped_id]
        pred_rg = sub['rg'].dropna().values
        # PED reference Rg from fig2 GT
        if len(fig2_conf) == 0: continue
        ref_rg = fig2_conf[fig2_conf['ped_id'] == ped_id]['rg_gt'].dropna().values
        if len(pred_rg) < 5 or len(ref_rg) < 5: continue
        ks_stat, p_val = ks_2samp(pred_rg, ref_rg)
        rows.append({
            'ped_id': ped_id,
            'KS_statistic': ks_stat, 'p_value': p_val,
            'pred_mean': pred_rg.mean(), 'pred_std': pred_rg.std(),
            'ref_mean': ref_rg.mean(), 'ref_std': ref_rg.std(),
        })
    ks_df = pd.DataFrame(rows)
    path = os.path.join(OUT, 'rg_ks_tests.csv')
    ks_df.to_csv(path, index=False)
    return ks_df

def build_three_condition_decomposition(cg_summary, fig2_summary):
    """Build three-condition decomposition table.

    CG+CODLAD: from cg_summary (this figure)
    PED+CODLAD: from fig2_summary pred columns
    PED ref: from fig2_summary gt columns
    """
    # Column mapping: (label, key, cg_col, fig2_pred_col, fig2_gt_col)
    metrics = [
        ('Rg (A)', 'rg', 'rg_mean', 'rg_pred_mean', 'rg_gt_mean'),
        ('Rama favored (%)', 'rama', 'rama_favored_pct', 'rama_favored_pct_pred', 'rama_favored_pct_gt'),
        ('Rota favored (%)', 'rota', 'rota_favored_pct', 'rota_favored_pct_pred', 'rota_favored_pct_gt'),
        ('Cbdev (A)', 'cbdev', 'cbdev_mean', 'cbdev_mean_pred_mean', 'cbdev_mean_gt_mean'),
        ('Clash', 'clash', 'clash_mean', 'clash_pred_mean', 'clash_gt_mean'),
        ('Helix fraction', 'helix', 'helix_frac', 'helix_frac_pred', 'helix_frac_gt'),
        ('Pro phi mean (deg)', 'pro_phi', 'pro_phi_mean', 'pro_phi_mean_pred', 'pro_phi_mean_gt'),
    ]

    rows = []
    for label, key, cg_col, fig2_pred_col, fig2_gt_col in metrics:
        cg_val = cg_summary[cg_col].mean() if cg_col in cg_summary.columns else np.nan
        fig2_pred_val = fig2_summary[fig2_pred_col].mean() if (len(fig2_summary) > 0 and fig2_pred_col in fig2_summary.columns) else np.nan
        fig2_gt_val = fig2_summary[fig2_gt_col].mean() if (len(fig2_summary) > 0 and fig2_gt_col in fig2_summary.columns) else np.nan

        total_delta = cg_val - fig2_gt_val if not (np.isnan(cg_val) or np.isnan(fig2_gt_val)) else np.nan
        pipeline_delta = fig2_pred_val - fig2_gt_val if not (np.isnan(fig2_pred_val) or np.isnan(fig2_gt_val)) else np.nan
        cg_delta = cg_val - fig2_pred_val if not (np.isnan(cg_val) or np.isnan(fig2_pred_val)) else np.nan

        rows.append({
            'metric': label, 'key': key,
            'PED_ref': fig2_gt_val, 'PED_CODLAD': fig2_pred_val, 'CG_CODLAD': cg_val,
            'total_delta': total_delta, 'pipeline_delta': pipeline_delta, 'CG_delta': cg_delta,
        })

    decomp = pd.DataFrame(rows)
    path = os.path.join(OUT, 'three_condition_decomposition.csv')
    decomp.to_csv(path, index=False)
    print(f'  Three-condition decomposition: {len(decomp)} metrics -> {path}')
    return decomp

def build_rama_three_way(cg_rama, fig2_rama_path):
    """Build three-way Ramachandran comparison by class."""
    rows = []
    fig2_rama = pd.read_csv(fig2_rama_path) if os.path.exists(fig2_rama_path) else pd.DataFrame()

    for cat in RAMA_CATS:
        # CG+CODLAD
        cg_sub = cg_rama[cg_rama['category'] == cat]
        cg_pct = cg_sub['favored_pct'].mean() if len(cg_sub) else np.nan

        # PED+CODLAD and PED ref from fig2
        ped_codlad_pct = np.nan
        ped_ref_pct = np.nan
        if len(fig2_rama) > 0:
            f2_sub = fig2_rama[fig2_rama['category'] == cat]
            if len(f2_sub) > 0:
                ped_codlad_pct = f2_sub['pred_favored_pct'].mean()
                ped_ref_pct = f2_sub['gt_favored_pct'].mean()

        rows.append({
            'category': cat,
            'PED_ref': ped_ref_pct, 'PED_CODLAD': ped_codlad_pct, 'CG_CODLAD': cg_pct,
        })

    rama3 = pd.DataFrame(rows)
    path = os.path.join(OUT, 'rama_three_way.csv')
    rama3.to_csv(path, index=False)
    return rama3

def build_report(cg_summary, fig2_summary, decomp, rama_df, rama3, bond_df, ca_df, ks_df):
    lines = [
        '# Figure 3 Step=100 Report: CG+CODLAD Pipeline',
        '',
        '## Configuration',
        '',
        '| Parameter | Value |',
        '|-----------|-------|',
        '| Pipeline | CALVADOS CA -> CVAE(C2) -> MPNN diffusion(100 steps) -> VAE(N6) decode |',
        '| Diffusion steps | 100 |',
        '| Frame sampling | Every 8th from 4000-frame trajectory (500 frames/system) |',
        '| Systems | 23 (clean_v5) |',
        '| Total frames | 11,500 |',
        '',
        '## Three-Condition Decomposition',
        '',
        'Total Δ = (CG + CODLAD) − (PED reference)',
        'Pipeline Δ = (PED + CODLAD) − (PED reference)',
        'CG Δ = (CG + CODLAD) − (PED + CODLAD)',
        '',
        '| Metric | PED ref | PED+CODLAD | CG+CODLAD | Total Δ | Pipeline Δ | CG Δ |',
        '|--------|---------|------------|-----------|---------|------------|------|',
    ]

    def fmt(v, d=3):
        if isinstance(v, float) and np.isnan(v): return '--'
        return f'{v:.{d}f}'

    for _, row in decomp.iterrows():
        lines.append(
            f'| {row["metric"]} | {fmt(row["PED_ref"])} | {fmt(row["PED_CODLAD"])} | '
            f'{fmt(row["CG_CODLAD"])} | {fmt(row["total_delta"])} | {fmt(row["pipeline_delta"])} | '
            f'{fmt(row["CG_delta"])} |')

    # Ramachandran three-way by class
    lines.extend(['', '## Ramachandran by Class (Three-Way)', ''])
    lines.append('| Class | PED ref | PED+CODLAD | CG+CODLAD | CG Δ |')
    lines.append('|-------|---------|------------|-----------|------|')
    if len(rama3) > 0:
        for _, row in rama3.iterrows():
            cg_d = row['CG_CODLAD'] - row['PED_CODLAD'] if not (np.isnan(row['CG_CODLAD']) or np.isnan(row['PED_CODLAD'])) else np.nan
            lines.append(f'| {row["category"]} | {fmt(row["PED_ref"], 1)} | {fmt(row["PED_CODLAD"], 1)} | {fmt(row["CG_CODLAD"], 1)} | {fmt(cg_d, 1)} |')

    # Bond geometry
    lines.extend(['', '## Bond Geometry', ''])
    canonical = {'CA_C': 1.53, 'N_CA': 1.46, 'C_N': 1.33}
    if len(bond_df) > 0:
        for bt in ['CA_C', 'N_CA', 'C_N']:
            sub = bond_df[bond_df['bond_type'] == bt]
            mean = sub['mean'].mean() if len(sub) else np.nan
            dev = mean - canonical[bt] if not np.isnan(mean) else np.nan
            lines.append(f'- {bt}: {fmt(mean, 4)} A (canonical {canonical[bt]:.2f}, dev {fmt(dev, 4)} A)')

    # CA fidelity
    lines.extend(['', '## CA Fidelity', ''])
    if len(ca_df) > 0:
        mean_dev = ca_df['mean_deviation'].mean()
        lines.append(f'- Mean CA deviation: {fmt(mean_dev, 6)} A (expected near zero)')

    # Rg KS tests
    lines.extend(['', '## Rg Distribution Comparison (KS tests)', ''])
    if len(ks_df) > 0:
        for _, row in ks_df.iterrows():
            sig = '*' if row['p_value'] < 0.05 else ''
            lines.append(f'- {row["ped_id"]}: KS={row["KS_statistic"]:.3f}, p={row["p_value"]:.3f}{sig}')

    # Per-system trans-Pro favored
    lines.extend(['', '## Per-System Trans-Pro Favored', ''])
    if len(rama_df) > 0:
        tp = rama_df[rama_df['category'] == 'trans_proline']
        for _, row in tp.iterrows():
            lines.append(f'- {row["ped_id"]}: {fmt(row["favored_pct"], 1)}%')

    # Interpretation
    lines.extend([
        '',
        '## Interpretation',
        '',
        '### Bond geometry and CA fidelity',
        'Expected unaffected by CG input. CODLAD reconstructs bond geometry from learned',
        'internal coordinates regardless of input source. CA positions are preserved exactly',
        'by the pipeline (CA fidelity ~0).',
        '',
        '### Rg and helix fraction',
        'Expected near PED reference. CALVADOS 3 samples broadly from the conformational',
        'ensemble, so Rg and secondary structure distributions should approximate PED.',
        '',
        '### Trans-proline Ramachandran',
        'Expected major CG-attributable degradation. CALVADOS coarse-grained force field',
        'does not model proline sidechain chirality, leading to trans-proline phi angle',
        'errors that propagate through CODLAD reconstruction.',
        '',
        '### Rotamer and clash',
        'Expected to track Figure 2 values (pipeline-attributable). These metrics depend',
        'on CODLAD internal coordinate reconstruction quality, not on input CA positions.',
    ])

    report = '\n'.join(lines)
    path = os.path.join(OUT, 'report.md')
    with open(path, 'w') as f:
        f.write(report)
    print(f'  Report: {path}')

def main():
    print('=' * 60)
    print('  Figure 3 Step=100 Merge')
    print('=' * 60)

    # Load per-frame metrics (prefer recomputed full CSV)
    per_frame_path = os.path.join(OUT, 'per_frame_metrics.csv')
    if os.path.exists(per_frame_path):
        df = pd.read_csv(per_frame_path)
        print(f'  Loaded {len(df)} rows from {per_frame_path}')
    else:
        df = load_cg_results()
    if len(df) == 0:
        print('  ERROR: No CG results found'); return
    print(f'  {len(df)} CG+CODLAD frames across {df["ped_id"].nunique()} systems')

    # Per-frame metrics already saved by recompute script
    print(f'  Per-frame: {len(df)} rows in {per_frame_path}')

    cg_summary = build_per_system_summary(df)
    rama_df = build_rama_by_class(df)
    bond_df = build_bond_geometry(df)
    ca_df = build_ca_fidelity(df)

    # Load Figure 2 data
    fig2_conf = load_fig2_per_conformer()
    fig2_summary = load_fig2_summary()
    print(f'  Figure 2: {len(fig2_conf)} conformers, {len(fig2_summary)} system summaries')

    # KS tests
    ks_df = build_rg_ks_tests(df, fig2_conf)

    # Three-condition decomposition
    decomp = build_three_condition_decomposition(cg_summary, fig2_summary)

    # Three-way Ramachandran
    fig2_rama_path = os.path.join(FIG2_OUT, 'rama_by_class.csv')
    rama3 = build_rama_three_way(rama_df, fig2_rama_path)

    # Report
    build_report(cg_summary, fig2_summary, decomp, rama_df, rama3, bond_df, ca_df, ks_df)
    print(f'\n  Merge complete.')

if __name__ == '__main__':
    main()
