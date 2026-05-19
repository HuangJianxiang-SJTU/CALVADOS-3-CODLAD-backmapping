"""Merge step=100 benchmark results into CSVs, report, and figures.

Usage:
  python scripts/fig2_step100_merge.py
"""
import os, sys, json, csv
import numpy as np
import pandas as pd

PROJECT = '/MDdata/data04/jxhuang/cg_cascade'
OUT = os.path.join(PROJECT, 'logs/figure2_step100')

# Load clean_v5 system list
with open(os.path.join(PROJECT, 'dataset-26-5-10/clean_v5.csv')) as f:
    CLEAN_V5 = sorted({r['ped_id'] for r in csv.DictReader(f)})

RAMA_CATS = ['general', 'glycine', 'pre_proline', 'trans_proline', 'cis_proline', 'ile_val']
AMINO_ACIDS = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
               'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']

def load_all_results():
    """Load all GPU result JSONs and filter to clean_v5 systems."""
    all_rows = []
    for gpu_id in range(4):
        path = os.path.join(OUT, f'results_gpu{gpu_id}.json')
        if not os.path.exists(path):
            continue
        with open(path) as f:
            rows = json.load(f)
        for r in rows:
            if r.get('ped_id') in CLEAN_V5:
                all_rows.append(r)
    return all_rows

def build_per_conformer_csv(rows):
    """Save per-conformer metrics CSV."""
    df = pd.DataFrame(rows)
    path = os.path.join(OUT, 'per_conformer_metrics.csv')
    df.to_csv(path, index=False)
    print(f'  Per-conformer: {len(df)} rows -> {path}')
    return df

def build_per_system_summary(df):
    """Build per-system summary with GT/Pred means, delta, rel_delta."""
    summary_rows = []
    for ped_id in sorted(df['ped_id'].unique()):
        sub = df[df['ped_id'] == ped_id]
        row = {'ped_id': ped_id, 'n_conformers': len(sub)}

        # Metrics to aggregate
        metrics = [
            ('rg', 'rg_gt', 'rg_pred'),
            ('dmax', 'dmax_gt', 'dmax_pred'),
            ('bb_rmsd', None, 'bb_rmsd'),
            ('sc_rmsd', None, 'sc_rmsd'),
            ('aa_rmsd', None, 'aa_rmsd'),
            ('sc_bb_ratio', None, 'sc_bb_ratio'),
            ('cbdev_mean', 'cbdev_mean_gt', 'cbdev_mean_pred'),
            ('clash', 'clash_gt', 'clash_pred'),
        ]

        for name, gt_col, pred_col in metrics:
            if gt_col and gt_col in sub.columns:
                gt_vals = sub[gt_col].dropna()
                row[f'{name}_gt_mean'] = gt_vals.mean() if len(gt_vals) else np.nan
                row[f'{name}_gt_std'] = gt_vals.std() if len(gt_vals) else np.nan
            else:
                row[f'{name}_gt_mean'] = np.nan
                row[f'{name}_gt_std'] = np.nan

            if pred_col and pred_col in sub.columns:
                pred_vals = sub[pred_col].dropna()
                row[f'{name}_pred_mean'] = pred_vals.mean() if len(pred_vals) else np.nan
                row[f'{name}_pred_std'] = pred_vals.std() if len(pred_vals) else np.nan
            else:
                row[f'{name}_pred_mean'] = np.nan
                row[f'{name}_pred_std'] = np.nan

            # Delta and relative delta
            gt_m = row.get(f'{name}_gt_mean', np.nan)
            pred_m = row.get(f'{name}_pred_mean', np.nan)
            if not np.isnan(gt_m) and not np.isnan(pred_m):
                row[f'{name}_delta'] = pred_m - gt_m
                row[f'{name}_rel_delta_pct'] = (pred_m - gt_m) / abs(gt_m) * 100 if abs(gt_m) > 1e-9 else np.nan
            else:
                row[f'{name}_delta'] = np.nan
                row[f'{name}_rel_delta_pct'] = np.nan

        # Ramachandran favored percentage
        for prefix in ['gt', 'pred']:
            fav = sub[f'rama_{prefix}_pooled_fav'].sum()
            tot = sub[f'rama_{prefix}_pooled_tot'].sum()
            row[f'rama_favored_pct_{prefix}'] = fav / tot * 100 if tot else np.nan

        # Rotamer favored percentage
        for prefix in ['gt', 'pred']:
            fav = sub[f'rota_fav_{prefix}'].sum()
            tot = sub[f'rota_tot_{prefix}'].sum()
            row[f'rota_favored_pct_{prefix}'] = fav / tot * 100 if tot else np.nan

        # Helix fraction
        for prefix in ['gt', 'pred']:
            h = sub[f'dssp_helix_{prefix}'].sum()
            t = sub[f'dssp_total_{prefix}'].sum()
            row[f'helix_frac_{prefix}'] = h / t if t else np.nan

        # Pro phi mean
        for prefix in ['gt', 'pred']:
            vals = sub[f'pro_phi_mean_{prefix}'].dropna()
            row[f'pro_phi_mean_{prefix}'] = vals.mean() if len(vals) else np.nan

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    path = os.path.join(OUT, 'per_system_summary.csv')
    summary.to_csv(path, index=False)
    print(f'  Per-system summary: {len(summary)} systems -> {path}')
    return summary

def build_rama_by_class(df):
    """Build Ramachandran by class CSV."""
    rows = []
    for ped_id in sorted(df['ped_id'].unique()):
        sub = df[df['ped_id'] == ped_id]
        for cat in RAMA_CATS:
            row = {'ped_id': ped_id, 'category': cat}
            for prefix in ['gt', 'pred']:
                fav = sub[f'rama_{prefix}_{cat}_fav'].sum()
                tot = sub[f'rama_{prefix}_{cat}_tot'].sum()
                row[f'{prefix}_favored'] = fav
                row[f'{prefix}_total'] = tot
                row[f'{prefix}_favored_pct'] = fav / tot * 100 if tot else np.nan
            rows.append(row)
    rama_df = pd.DataFrame(rows)
    path = os.path.join(OUT, 'rama_by_class.csv')
    rama_df.to_csv(path, index=False)
    print(f'  Ramachandran by class: {len(rama_df)} rows -> {path}')
    return rama_df

def build_per_residue_type_sc_rmsd(df):
    """Build per-residue-type sidechain RMSD CSV."""
    # Collect per-conformer per-residue-type SC RMSD
    sc_cols = [c for c in df.columns if c.startswith('sc_rmsd_') and c != 'sc_rmsd']
    if not sc_cols:
        print('  WARNING: No per-residue-type SC RMSD columns found')
        return pd.DataFrame()

    rows = []
    for col in sc_cols:
        res3 = col.replace('sc_rmsd_', '')
        vals = df[col].dropna()
        if len(vals) == 0: continue
        rows.append({
            'amino_acid': res3,
            'mean_rmsd': vals.mean(),
            'std_rmsd': vals.std(),
            'median_rmsd': vals.median(),
            'n': len(vals),
        })

    result = pd.DataFrame(rows).sort_values('mean_rmsd')
    path = os.path.join(OUT, 'per_residue_type_sc_rmsd.csv')
    result.to_csv(path, index=False)
    print(f'  Per-residue-type SC RMSD: {len(result)} types -> {path}')
    return result

def build_bond_geometry(df):
    """Build bond geometry CSV."""
    rows = []
    for ped_id in sorted(df['ped_id'].unique()):
        sub = df[df['ped_id'] == ped_id]
        for bt in ['CA_C', 'N_CA', 'C_N']:
            row = {'ped_id': ped_id, 'bond_type': bt}
            for prefix in ['gt', 'pred']:
                mean_col = f'bond_{prefix}_mean_{bt}'
                std_col = f'bond_{prefix}_std_{bt}'
                if mean_col in sub.columns:
                    vals = sub[mean_col].dropna()
                    row[f'{prefix}_mean'] = vals.mean() if len(vals) else np.nan
                if std_col in sub.columns:
                    vals = sub[std_col].dropna()
                    row[f'{prefix}_std'] = vals.mean() if len(vals) else np.nan
            rows.append(row)
    bond_df = pd.DataFrame(rows)
    path = os.path.join(OUT, 'bond_geometry.csv')
    bond_df.to_csv(path, index=False)
    print(f'  Bond geometry: {len(bond_df)} rows -> {path}')
    return bond_df

def build_panel_csvs(df):
    """Save per-panel CSVs for figure plotting."""
    # Panel A: Rg pairwise
    rg_data = df[['ped_id', 'conformer_idx', 'rg_gt', 'rg_pred']].dropna()
    rg_data.to_csv(os.path.join(OUT, 'figure2_panelA_rg_pairwise.csv'), index=False)

    # Panel B: Dmax pairwise
    dmax_data = df[['ped_id', 'conformer_idx', 'dmax_gt', 'dmax_pred']].dropna()
    dmax_data.to_csv(os.path.join(OUT, 'figure2_panelB_dmax_pairwise.csv'), index=False)

    # Panel E: Per-residue-type SC RMSD
    sc_cols = [c for c in df.columns if c.startswith('sc_rmsd_') and c != 'sc_rmsd']
    if sc_cols:
        sc_data = df[['ped_id', 'conformer_idx'] + sc_cols].copy()
        sc_data.to_csv(os.path.join(OUT, 'figure2_panelE_sc_rmsd_by_type.csv'), index=False)

    # Panel F: Clash per system
    clash_data = df[['ped_id', 'conformer_idx', 'clash_gt', 'clash_pred']].dropna()
    clash_data.to_csv(os.path.join(OUT, 'figure2_panelF_clash.csv'), index=False)

    # Supplementary panels
    supp_cols = ['ped_id', 'conformer_idx', 'bb_rmsd', 'sc_rmsd', 'aa_rmsd', 'sc_bb_ratio',
                 'cbdev_mean_gt', 'cbdev_mean_pred', 'dssp_helix_gt', 'dssp_total_gt',
                 'dssp_helix_pred', 'dssp_total_pred']
    avail_cols = [c for c in supp_cols if c in df.columns]
    df[avail_cols].to_csv(os.path.join(OUT, 'figure2_supp_data.csv'), index=False)

    print(f'  Panel CSVs saved')

def build_report(df, summary, rama_df, sc_type_df, bond_df):
    """Build comprehensive report.md."""
    lines = [
        '# Figure 2 Step=100 Benchmark Report',
        '',
        '## Configuration',
        '',
        '| Parameter | Value |',
        '|-----------|-------|',
        '| Pipeline | CVAE(C2) -> MPNN diffusion(100 steps) -> VAE(N6) decode |',
        '| Diffusion steps | 100 (author default) |',
        '| Conformer cap | 500 per system (evenly-spaced striding) |',
        f'| Total conformers | {len(df)} |',
        f'| Systems | {len(summary)} (clean_v5) |',
        '',
    ]

    # Per-system conformer counts
    lines.extend([
        '### Conformer counts per system',
        '',
        '| PED ID | N conformers |',
        '|--------|-------------|',
    ])
    for _, row in summary.iterrows():
        lines.append(f'| {row["ped_id"]} | {int(row["n_conformers"])} |')

    # Aggregate metrics table
    lines.extend([
        '',
        '## Aggregate Metrics',
        '',
        '| Metric | GT mean +/- std | Pred mean +/- std | Delta | Rel delta (%) |',
        '|--------|----------------|-------------------|-------|---------------|',
    ])

    agg_metrics = [
        ('Rg (A)', 'rg'),
        ('Dmax (A)', 'dmax'),
        ('BB RMSD (A)', 'bb_rmsd'),
        ('SC RMSD (A)', 'sc_rmsd'),
        ('AA RMSD (A)', 'aa_rmsd'),
        ('SC/BB ratio', 'sc_bb_ratio'),
        ('Cbdev mean (A)', 'cbdev_mean'),
        ('Clash count', 'clash'),
    ]

    for label, name in agg_metrics:
        gt_m = summary[f'{name}_gt_mean'].mean() if f'{name}_gt_mean' in summary.columns else np.nan
        gt_s = summary[f'{name}_gt_std'].mean() if f'{name}_gt_std' in summary.columns else np.nan
        pred_m = summary[f'{name}_pred_mean'].mean() if f'{name}_pred_mean' in summary.columns else np.nan
        pred_s = summary[f'{name}_pred_std'].mean() if f'{name}_pred_std' in summary.columns else np.nan
        delta = summary[f'{name}_delta'].mean() if f'{name}_delta' in summary.columns else np.nan
        rel = summary[f'{name}_rel_delta_pct'].mean() if f'{name}_rel_delta_pct' in summary.columns else np.nan

        def fmt(v, d=3): return f'{v:.{d}f}' if not np.isnan(v) else '--'
        lines.append(f'| {label} | {fmt(gt_m)} +/- {fmt(gt_s)} | {fmt(pred_m)} +/- {fmt(pred_s)} | {fmt(delta)} | {fmt(rel, 1)} |')

    # Ramachandran favored
    rama_gt_pct = summary['rama_favored_pct_gt'].mean() if 'rama_favored_pct_gt' in summary.columns else np.nan
    rama_pred_pct = summary['rama_favored_pct_pred'].mean() if 'rama_favored_pct_pred' in summary.columns else np.nan
    lines.append(f'| Rama favored (%) | {fmt(rama_gt_pct, 1)} | {fmt(rama_pred_pct, 1)} | {fmt(rama_pred_pct - rama_gt_pct, 1)} | -- |')

    rota_gt_pct = summary['rota_favored_pct_gt'].mean() if 'rota_favored_pct_gt' in summary.columns else np.nan
    rota_pred_pct = summary['rota_favored_pct_pred'].mean() if 'rota_favored_pct_pred' in summary.columns else np.nan
    lines.append(f'| Rota favored (%) | {fmt(rota_gt_pct, 1)} | {fmt(rota_pred_pct, 1)} | {fmt(rota_pred_pct - rota_gt_pct, 1)} | -- |')

    # Ramachandran by class
    lines.extend([
        '',
        '## Ramachandran by Class',
        '',
        '| Category | GT favored (%) | Pred favored (%) | Delta (%) |',
        '|----------|---------------|-----------------|-----------|',
    ])
    for cat in RAMA_CATS:
        sub = rama_df[rama_df['category'] == cat]
        gt_pct = sub['gt_favored_pct'].mean() if len(sub) else np.nan
        pred_pct = sub['pred_favored_pct'].mean() if len(sub) else np.nan
        delta = pred_pct - gt_pct if not (np.isnan(gt_pct) or np.isnan(pred_pct)) else np.nan
        lines.append(f'| {cat} | {fmt(gt_pct, 1)} | {fmt(pred_pct, 1)} | {fmt(delta, 1)} |')

    # Per-residue-type SC RMSD ranking
    if len(sc_type_df) > 0:
        lines.extend([
            '',
            '## Per-Residue-Type Sidechain RMSD (ranked best-to-worst)',
            '',
            '| Amino acid | Mean RMSD (A) | Std | N |',
            '|------------|--------------|-----|---|',
        ])
        for _, row in sc_type_df.iterrows():
            lines.append(f'| {row["amino_acid"]} | {row["mean_rmsd"]:.3f} | {row["std_rmsd"]:.3f} | {int(row["n"])} |')

    # Per-system breakdown
    lines.extend([
        '',
        '## Per-System Breakdown',
        '',
        '| PED ID | N conf | Rg GT | Rg Pred | Rama GT% | Rama Pred% | Clash GT | Clash Pred |',
        '|--------|--------|-------|---------|----------|------------|----------|------------|',
    ])
    for _, row in summary.iterrows():
        rg_gt = row.get('rg_gt_mean', np.nan)
        rg_pred = row.get('rg_pred_mean', np.nan)
        rama_gt = row.get('rama_favored_pct_gt', np.nan)
        rama_pred = row.get('rama_favored_pct_pred', np.nan)
        clash_gt = row.get('clash_gt_mean', np.nan)
        clash_pred = row.get('clash_pred_mean', np.nan)
        lines.append(f'| {row["ped_id"]} | {int(row["n_conformers"])} | {fmt(rg_gt)} | {fmt(rg_pred)} | {fmt(rama_gt,1)} | {fmt(rama_pred,1)} | {fmt(clash_gt,1)} | {fmt(clash_pred,1)} |')

    # Bond geometry
    if len(bond_df) > 0:
        lines.extend([
            '',
            '## Bond Geometry',
            '',
            '| Bond type | GT mean (A) | Pred mean (A) | Delta (A) |',
            '|-----------|------------|--------------|----------|',
        ])
        for bt in ['CA_C', 'N_CA', 'C_N']:
            sub = bond_df[bond_df['bond_type'] == bt]
            gt_m = sub['gt_mean'].mean() if len(sub) else np.nan
            pred_m = sub['pred_mean'].mean() if len(sub) else np.nan
            delta = pred_m - gt_m if not (np.isnan(gt_m) or np.isnan(pred_m)) else np.nan
            lines.append(f'| {bt} | {fmt(gt_m, 4)} | {fmt(pred_m, 4)} | {fmt(delta, 4)} |')

    # Decomposition note
    lines.extend([
        '',
        '## Decomposition Note',
        '',
        'This figure establishes the **pipeline upper bound** for CODLAD reconstruction',
        'fidelity when experimentally-derived PDB conformers are supplied directly as input.',
        'The metrics reported here represent the best achievable quality of the',
        'CVAE(C2) -> MPNN diffusion(100 steps) -> VAE(N6) decode pipeline.',
        '',
        'Any further degradation observed in Figure 3 (which uses CALVADOS CG input',
        'instead of PDB input) is attributable to the CG input quality, not the pipeline itself.',
        'The decomposition is:',
        '',
        '  Total error = Pipeline error (this figure) + CG input error (Figure 3)',
        '',
        'Step=100 is the author-default diffusion configuration, providing finer-grained',
        'denoising compared to step=5 used in preliminary experiments.',
    ])

    report = '\n'.join(lines)
    path = os.path.join(OUT, 'report.md')
    with open(path, 'w') as f:
        f.write(report)
    print(f'  Report: {path}')

def main():
    print('=' * 60)
    print('  Figure 2 Step=100 Merge')
    print('=' * 60)

    # Load results
    rows = load_all_results()
    if not rows:
        print('  ERROR: No results found')
        return
    print(f'  Loaded {len(rows)} conformer results')

    # Build outputs
    df = build_per_conformer_csv(rows)
    summary = build_per_system_summary(df)
    rama_df = build_rama_by_class(df)
    sc_type_df = build_per_residue_type_sc_rmsd(df)
    bond_df = build_bond_geometry(df)
    build_panel_csvs(df)
    build_report(df, summary, rama_df, sc_type_df, bond_df)

    print(f'\n  Merge complete.')

if __name__ == '__main__':
    main()
