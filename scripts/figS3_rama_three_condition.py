#!/usr/bin/env python3
"""Figure S3: Three-condition six-class Ramachandran validation figure.

3 rows (PED reference, PED+CODLAD, CG+CODLAD) x 6 columns (residue classes).
Each panel shows MolProbity contour background + (phi,psi) scatter points
colored by classification (favored=black, allowed=gray, outlier=red).
"""
import os, sys, warnings, time, pickle, io
warnings.filterwarnings("ignore")

os.chdir('/MDdata/data04/jxhuang/cg_cascade')
sys.path.insert(0, 'src/cascade_codlad/eval_ensemble')

import numpy as np, pandas as pd
import mdtraj as md
from collections import defaultdict
from metrics_ramachandran import _in_boxes, _get_region_boxes

PROJECT = '/MDdata/data04/jxhuang/cg_cascade'
OUT = os.path.join(PROJECT, 'logs/figure3_step100')
FIG2_CACHE = os.path.join(PROJECT, 'logs/figure2_step100/cache')
FIG3_CACHE = os.path.join(PROJECT, 'logs/figure3_step100/cache')
CONTOUR_DIR = os.path.join(PROJECT, 'a_rama')

os.makedirs(OUT, exist_ok=True)

# The 23 systems in clean_v5
SYSTEMS = sorted([
    'PED00003', 'PED00006', 'PED00016', 'PED00074', 'PED00140', 'PED00141',
    'PED00154', 'PED00155', 'PED00181', 'PED00183', 'PED00184', 'PED00229',
    'PED00231', 'PED00233', 'PED00238', 'PED00442', 'PED00468', 'PED00483',
    'PED00484', 'PED00493', 'PED00494', 'PED00495', 'PED00536',
])

RESIDUE_CLASSES = ['General', 'Glycine', 'Ile/Val', 'Pre-Pro', 'Trans-Pro', 'Cis-Pro']
CLASS_MAP = {
    'general': 'General', 'glycine': 'Glycine', 'ile_val': 'Ile/Val',
    'pre_proline': 'Pre-Pro', 'trans_proline': 'Trans-Pro', 'cis_proline': 'Cis-Pro',
}
CONTOUR_INDEX = {'General': 0, 'Glycine': 1, 'Pre-Pro': 2, 'Ile/Val': 3, 'Trans-Pro': 4, 'Cis-Pro': 5}

# ── Compute (phi,psi) from cache ──────────────────────────────────────────────
def compute_rama_points(xyz, top):
    """Compute per-residue (phi, psi, category, classification) for one frame."""
    traj = md.Trajectory(xyz[np.newaxis, ...] / 10.0, topology=top)
    phi_d = np.degrees(md.compute_phi(traj)[1][0])
    psi_d = np.degrees(md.compute_psi(traj)[1][0])
    omega_d = np.degrees(md.compute_omega(traj)[1][0])
    rn = [r.name for r in top.residues]

    rows = []
    for i, res3 in enumerate(rn):
        nxt = rn[i+1] if i+1 < len(rn) else 'XXX'
        p = float(phi_d[i-1]) if 0 < i < len(phi_d)+1 else np.nan
        s = float(psi_d[i]) if i < len(psi_d) else np.nan
        o = float(omega_d[i-1]) if 0 < i < len(omega_d)+1 else np.nan
        if np.isnan(p) or np.isnan(s):
            continue

        # Classify
        if res3 == 'GLY': cat = 'glycine'
        elif res3 == 'PRO': cat = 'cis_proline' if (not np.isnan(o) and abs(o) < 90) else 'trans_proline'
        elif nxt == 'PRO': cat = 'pre_proline'
        elif res3 in ('ILE', 'VAL'): cat = 'ile_val'
        else: cat = 'general'

        result = _get_region_boxes(res3, nxt)
        if result:
            fb, ab = result
            cls = 'favored' if _in_boxes(p, s, fb) else ('allowed' if _in_boxes(p, s, ab) else 'outlier')
        else:
            cls = 'unknown'

        rows.append({'phi': p, 'psi': s, 'category': CLASS_MAP.get(cat, cat), 'classification': cls})
    return rows

# ── Collect points for each condition ─────────────────────────────────────────
def collect_fig2_points():
    """Collect (phi,psi) from fig2 cache for PED reference and PED+CODLAD."""
    gt_points = []   # PED reference
    pred_points = [] # PED+CODLAD

    cache_files = sorted([f for f in os.listdir(FIG2_CACHE) if f.endswith('.npz')])
    print(f'Fig2 cache: {len(cache_files)} frames', flush=True)

    t0 = time.time()
    for i, npz_file in enumerate(cache_files):
        key = npz_file[:-4]
        ped_id = key.rsplit('_', 1)[0]
        if ped_id not in SYSTEMS:
            continue

        data = np.load(os.path.join(FIG2_CACHE, npz_file), allow_pickle=True)
        top_path = os.path.join(FIG2_CACHE, f'{key}_top.pkl')
        with open(top_path, 'rb') as f:
            top = pickle.load(f)

        # GT (reference)
        if 'xyz_gt' in data:
            gt_rows = compute_rama_points(data['xyz_gt'], top)
            gt_points.extend(gt_rows)

        # Pred (PED+CODLAD)
        if 'xyz_pred' in data:
            pred_rows = compute_rama_points(data['xyz_pred'], top)
            pred_points.extend(pred_rows)

        if (i+1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i+1) / elapsed
            print(f'  Fig2 [{i+1}/{len(cache_files)}] {rate:.1f}/s', flush=True)

    print(f'  GT points: {len(gt_points)}, Pred points: {len(pred_points)}', flush=True)
    return gt_points, pred_points

def collect_fig3_points():
    """Collect (phi,psi) from fig3 cache for CG+CODLAD."""
    points = []

    cache_files = sorted([f for f in os.listdir(FIG3_CACHE) if f.endswith('.npz')])
    print(f'Fig3 cache: {len(cache_files)} frames', flush=True)

    t0 = time.time()
    for i, npz_file in enumerate(cache_files):
        key = npz_file[:-4]
        ped_id = key.rsplit('_', 1)[0]
        if ped_id not in SYSTEMS:
            continue

        data = np.load(os.path.join(FIG3_CACHE, npz_file), allow_pickle=True)
        top_path = os.path.join(FIG3_CACHE, f'{key}_top.pkl')
        with open(top_path, 'rb') as f:
            top = pickle.load(f)

        rows = compute_rama_points(data['xyz_recon'], top)
        points.extend(rows)

        if (i+1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i+1) / elapsed
            print(f'  Fig3 [{i+1}/{len(cache_files)}] {rate:.1f}/s', flush=True)

    print(f'  CG+CODLAD points: {len(points)}', flush=True)
    return points

# ── Main computation ──────────────────────────────────────────────────────────
print('=' * 60)
print('  Figure S3: Three-condition Ramachandran')
print('=' * 60)
t_total = time.time()

# Check for cached data
cache_path = os.path.join(OUT, 'rama_S3_points.npz')
if os.path.exists(cache_path):
    print('Loading cached points...', flush=True)
    cached = np.load(cache_path, allow_pickle=True)
    gt_points = list(cached['gt'])
    pred_points = list(cached['pred'])
    cg_points = list(cached['cg'])
    print(f'  GT: {len(gt_points)}, Pred: {len(pred_points)}, CG: {len(cg_points)}', flush=True)
else:
    print('Computing Fig2 points (PED ref + PED+CODLAD)...', flush=True)
    gt_points, pred_points = collect_fig2_points()

    print('Computing Fig3 points (CG+CODLAD)...', flush=True)
    cg_points = collect_fig3_points()

    # Cache for reuse
    np.savez(cache_path, gt=gt_points, pred=pred_points, cg=cg_points)
    print(f'Cached points to {cache_path}', flush=True)

# Convert to DataFrames
df_gt = pd.DataFrame(gt_points)
df_pred = pd.DataFrame(pred_points)
df_cg = pd.DataFrame(cg_points)

conditions = [
    ('PED reference', df_gt),
    ('PED+CODLAD', df_pred),
    ('CG+CODLAD', df_cg),
]

# ── Generate outlier counts CSV ───────────────────────────────────────────────
print('\nComputing outlier counts...', flush=True)
csv_rows = []
for cond_name, df in conditions:
    for cls in RESIDUE_CLASSES:
        sub = df[df['category'] == cls]
        n_total = len(sub)
        n_favored = (sub['classification'] == 'favored').sum()
        n_allowed = (sub['classification'] == 'allowed').sum()
        n_outlier = (sub['classification'] == 'outlier').sum()
        pct_outlier = 100.0 * n_outlier / n_total if n_total > 0 else 0
        csv_rows.append({
            'condition': cond_name, 'residue_class': cls,
            'n_total': n_total, 'n_favored': n_favored,
            'n_allowed': n_allowed, 'n_outlier': n_outlier,
            'pct_outlier': pct_outlier,
        })

df_csv = pd.DataFrame(csv_rows)
csv_path = os.path.join(OUT, 'rama_S3_outlier_counts.csv')
df_csv.to_csv(csv_path, index=False)
print(f'Saved {csv_path}', flush=True)
print(df_csv.to_string(index=False))

# ── Sanity check ──────────────────────────────────────────────────────────────
print('\nSanity check (PED ref outlier% vs 100-favored%):', flush=True)
rama_by_class = os.path.join(PROJECT, 'logs/figure2_step100/rama_by_class.csv')
if os.path.exists(rama_by_class):
    df_rbc = pd.read_csv(rama_by_class)
    for cls in RESIDUE_CLASSES:
        cat_key = {'General':'general','Glycine':'glycine','Ile/Val':'ile_val',
                   'Pre-Pro':'pre_proline','Trans-Pro':'trans_proline','Cis-Pro':'cis_proline'}[cls]
        sub = df_rbc[df_rbc['category'] == cat_key]
        if len(sub) > 0:
            favored_pct = sub['gt_favored_pct'].mean()
            outlier_row = df_csv[(df_csv['condition']=='PED reference') & (df_csv['residue_class']==cls)]
            outlier_pct = outlier_row.iloc[0]['pct_outlier'] if len(outlier_row) > 0 else 0
            print(f'  {cls}: favored={favored_pct:.1f}%, outlier={outlier_pct:.1f}%, '
                  f'expected_outlier~{100-favored_pct:.1f}%', flush=True)

# ── Generate figure ───────────────────────────────────────────────────────────
print('\nGenerating figure...', flush=True)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 18, "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14,
    "figure.dpi": 600, "savefig.dpi": 600, "mathtext.default": "regular",
})

MAX_POINTS = 5000  # subsample per panel
np.random.seed(42)

fig, axes = plt.subplots(3, 6, figsize=(24, 12))

for row_idx, (cond_name, df) in enumerate(conditions):
    for col_idx, cls in enumerate(RESIDUE_CLASSES):
        ax = axes[row_idx, col_idx]
        sub = df[df['category'] == cls]

        # Subsample
        n_total = len(sub)
        if n_total > MAX_POINTS:
            idx = np.random.choice(n_total, MAX_POINTS, replace=False)
            sub_plot = sub.iloc[idx]
        else:
            sub_plot = sub

        # Plot points by classification
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

        # Outlier percentage annotation
        n_out = (sub['classification'] == 'outlier').sum()
        pct_out = 100.0 * n_out / n_total if n_total > 0 else 0
        ax.text(0.03, 0.97, f'outliers: {pct_out:.1f}%', transform=ax.transAxes,
                fontsize=9, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        # Crosshairs
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)

        # Axes
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_xticks([-90, 0, 90])
        ax.set_yticks([-90, 0, 90])

        # Spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Column titles (top row only)
        if row_idx == 0:
            ax.set_title(cls, fontsize=14, fontweight='bold')

        # Row labels (leftmost column)
        if col_idx == 0:
            row_label = ['S3A', 'S3B', 'S3C'][row_idx]
            ax.set_ylabel(f'{row_label}\n{cond_name}\nψ (°)', fontsize=12)

        # Axis labels
        if row_idx == 2:
            ax.set_xlabel('φ (°)')
        if col_idx > 0:
            ax.set_yticklabels([])

fig.tight_layout()

# Save PNG and PDF
png_path = os.path.join(OUT, 'figure_S3_rama_six_panel_three_condition.png')
pdf_path = os.path.join(OUT, 'figure_S3_rama_six_panel_three_condition.pdf')
fig.savefig(png_path, bbox_inches='tight', dpi=600)
fig.savefig(pdf_path, bbox_inches='tight', dpi=600)
plt.close(fig)

print(f'\nSaved: {png_path}')
print(f'Saved: {pdf_path}')
print(f'\nTotal time: {(time.time()-t_total)/60:.1f} min')
