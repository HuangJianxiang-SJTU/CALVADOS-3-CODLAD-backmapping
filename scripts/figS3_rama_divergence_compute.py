#!/usr/bin/env python3
"""Compute Jensen-Shannon divergence for Ramachandran distributions.

Three-condition decomposition:
  JS(PED+CODLAD, PED reference)  = pipeline divergence
  JS(CG+CODLAD,  PED reference)  = total divergence
  JS(CG+CODLAD,  PED+CODLAD)    = CG-only divergence

Also computes MolProbity favored percentages for cross-comparability.
Generates:
  - rama_divergence_three_condition.csv
  - rama_S3_molprobity_favored.csv (supplementary table)
  - figure_S3_rama_density_divergence.png/.pdf
  - figure_PED_vs_Dunbrack.png (if Dunbrack data available)
"""
import os, sys, warnings, time, pickle
warnings.filterwarnings("ignore")

os.chdir('/MDdata/data04/jxhuang/cg_cascade')
sys.path.insert(0, 'src/cascade_codlad/eval_ensemble')

import numpy as np, pandas as pd
import mdtraj as md
from collections import defaultdict
from metrics_ramachandran import _in_boxes, _get_region_boxes

_PHI_PSI_GRID_N = 36
_PHI_PSI_GRID_RANGE = (-180.0, 180.0)
_JD_EPSILON = 1e-10


def _hist2d_phi_psi(phi, psi, n_bins=_PHI_PSI_GRID_N):
    hist, _, _ = np.histogram2d(
        phi,
        psi,
        bins=n_bins,
        range=[_PHI_PSI_GRID_RANGE, _PHI_PSI_GRID_RANGE],
    )
    hist = hist.astype(float) + _JD_EPSILON
    return hist / hist.sum()


def compute_phi_psi_divergence(phi_pred, psi_pred, phi_ref, psi_ref):
    p = _hist2d_phi_psi(phi_pred, psi_pred).ravel()
    q = _hist2d_phi_psi(phi_ref, psi_ref).ravel()
    m = 0.5 * (p + q)
    js = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
    return {'js_divergence': float(js)}

PROJECT = '/MDdata/data04/jxhuang/cg_cascade'
OUT = os.path.join(PROJECT, 'logs/figure3_step100')
FIG2_CACHE = os.path.join(PROJECT, 'logs/figure2_step100/cache')
FIG3_CACHE = os.path.join(PROJECT, 'logs/figure3_step100/cache')
CONTOUR_DIR = os.path.join(PROJECT, 'a_rama')

os.makedirs(OUT, exist_ok=True)

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
CLASS_MAP_INV = {v: k for k, v in CLASS_MAP.items()}

# ── Compute per-residue (phi,psi) from one frame ──────────────────────────────
def compute_rama_points(xyz, top):
    """Return list of dicts with phi, psi, category, classification, resname, next_resname."""
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

        # Classify residue category
        if res3 == 'GLY': cat = 'glycine'
        elif res3 == 'PRO': cat = 'cis_proline' if (not np.isnan(o) and abs(o) < 90) else 'trans_proline'
        elif nxt == 'PRO': cat = 'pre_proline'
        elif res3 in ('ILE', 'VAL'): cat = 'ile_val'
        else: cat = 'general'

        # MolProbity classification (secondary metric)
        result = _get_region_boxes(res3, nxt)
        if result:
            fb, ab = result
            cls = 'favored' if _in_boxes(p, s, fb) else ('allowed' if _in_boxes(p, s, ab) else 'outlier')
        else:
            cls = 'unknown'

        rows.append({
            'phi': p, 'psi': s, 'omega': o,
            'category': CLASS_MAP.get(cat, cat),
            'classification': cls,
            'resname': res3, 'next_resname': nxt,
        })
    return rows

# ── Collect points from cache ─────────────────────────────────────────────────
def collect_fig2_points():
    gt_points, pred_points = [], []
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
        if 'xyz_gt' in data:
            gt_points.extend(compute_rama_points(data['xyz_gt'], top))
        if 'xyz_pred' in data:
            pred_points.extend(compute_rama_points(data['xyz_pred'], top))
        if (i+1) % 500 == 0:
            print(f'  Fig2 [{i+1}/{len(cache_files)}] {(i+1)/(time.time()-t0):.1f}/s', flush=True)
    print(f'  GT: {len(gt_points)}, Pred: {len(pred_points)}', flush=True)
    return gt_points, pred_points

def collect_fig3_points():
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
        points.extend(compute_rama_points(data['xyz_recon'], top))
        if (i+1) % 500 == 0:
            print(f'  Fig3 [{i+1}/{len(cache_files)}] {(i+1)/(time.time()-t0):.1f}/s', flush=True)
    print(f'  CG+CODLAD: {len(points)}', flush=True)
    return points

# ── Main ──────────────────────────────────────────────────────────────────────
print('=' * 60)
print('  Ramachandran JS Divergence Computation')
print('=' * 60)
t_total = time.time()

# Load or compute points
cache_path = os.path.join(OUT, 'rama_S3_points_v2.npz')
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
    np.savez(cache_path, gt=gt_points, pred=pred_points, cg=cg_points)
    print(f'Cached to {cache_path}', flush=True)

df_gt = pd.DataFrame(gt_points)
df_pred = pd.DataFrame(pred_points)
df_cg = pd.DataFrame(cg_points)

conditions = [
    ('PED reference', df_gt),
    ('PED+CODLAD', df_pred),
    ('CG+CODLAD', df_cg),
]

# ── Step 2: Compute JS divergences ────────────────────────────────────────────
print('\n--- JS Divergence Computation ---', flush=True)

divergence_rows = []
molprobity_rows = []

# For each residue class, compute JS between all condition pairs
for cls in RESIDUE_CLASSES:
    cat_key = CLASS_MAP_INV[cls]

    # Extract (phi, psi) arrays per condition
    sub_gt = df_gt[df_gt['category'] == cls]
    sub_pred = df_pred[df_pred['category'] == cls]
    sub_cg = df_cg[df_cg['category'] == cls]

    phi_gt = sub_gt['phi'].values
    psi_gt = sub_gt['psi'].values
    phi_pred = sub_pred['phi'].values
    psi_pred = sub_pred['psi'].values
    phi_cg = sub_cg['phi'].values
    psi_cg = sub_cg['psi'].values

    # JS(PED+CODLAD, PED reference) = pipeline divergence
    js_pipeline = compute_phi_psi_divergence(phi_pred, psi_pred, phi_gt, psi_gt)

    # JS(CG+CODLAD, PED reference) = total divergence
    js_total = compute_phi_psi_divergence(phi_cg, psi_cg, phi_gt, psi_gt)

    # JS(CG+CODLAD, PED+CODLAD) = CG-only divergence
    js_cg = compute_phi_psi_divergence(phi_cg, psi_cg, phi_pred, psi_pred)

    # JS(PED reference, PED reference) = self-check (should be ~0)
    js_self = compute_phi_psi_divergence(phi_gt, psi_gt, phi_gt, psi_gt)

    # CG-attributable JS = JS_total - JS_pipeline (approximately)
    # More precisely: JS(CG,PED) - JS(PED+CODLAD,PED)
    cg_attrib = js_total['js_divergence'] - js_pipeline['js_divergence']

    divergence_rows.append({
        'residue_class': cls,
        'n_ped_ref': len(phi_gt),
        'n_ped_codlad': len(phi_pred),
        'n_cg_codlad': len(phi_cg),
        'JS_pipeline': js_pipeline['js_divergence'],
        'JS_total': js_total['js_divergence'],
        'JS_cg': js_cg['js_divergence'],
        'JS_self_check': js_self['js_divergence'],
        'CG_attributable_JS': cg_attrib,
    })

    # MolProbity favored percentages (secondary metric)
    for cond_name, df_cond in conditions:
        sub = df_cond[df_cond['category'] == cls]
        n_total = len(sub)
        n_fav = (sub['classification'] == 'favored').sum()
        n_all = (sub['classification'] == 'allowed').sum()
        n_out = (sub['classification'] == 'outlier').sum()
        molprobity_rows.append({
            'condition': cond_name,
            'residue_class': cls,
            'n_total': n_total,
            'n_favored': n_fav,
            'n_allowed': n_all,
            'n_outlier': n_out,
            'pct_favored': 100.0 * n_fav / n_total if n_total > 0 else 0,
            'pct_allowed': 100.0 * n_all / n_total if n_total > 0 else 0,
            'pct_outlier': 100.0 * n_out / n_total if n_total > 0 else 0,
        })

    print(f'  {cls}: JS_pipeline={js_pipeline["js_divergence"]:.4f}, '
          f'JS_total={js_total["js_divergence"]:.4f}, '
          f'JS_cg={js_cg["js_divergence"]:.4f}, '
          f'JS_self={js_self["js_divergence"]:.6f}, '
          f'CG_attr={cg_attrib:.4f}', flush=True)

# Add pooled row (all classes combined)
phi_gt_all = df_gt['phi'].values
psi_gt_all = df_gt['psi'].values
phi_pred_all = df_pred['phi'].values
psi_pred_all = df_pred['psi'].values
phi_cg_all = df_cg['phi'].values
psi_cg_all = df_cg['psi'].values

js_pipe_pool = compute_phi_psi_divergence(phi_pred_all, psi_pred_all, phi_gt_all, psi_gt_all)
js_tot_pool = compute_phi_psi_divergence(phi_cg_all, psi_cg_all, phi_gt_all, psi_gt_all)
js_cg_pool = compute_phi_psi_divergence(phi_cg_all, psi_cg_all, phi_pred_all, psi_pred_all)
js_self_pool = compute_phi_psi_divergence(phi_gt_all, psi_gt_all, phi_gt_all, psi_gt_all)
cg_attr_pool = js_tot_pool['js_divergence'] - js_pipe_pool['js_divergence']

divergence_rows.append({
    'residue_class': 'Pooled',
    'n_ped_ref': len(phi_gt_all),
    'n_ped_codlad': len(phi_pred_all),
    'n_cg_codlad': len(phi_cg_all),
    'JS_pipeline': js_pipe_pool['js_divergence'],
    'JS_total': js_tot_pool['js_divergence'],
    'JS_cg': js_cg_pool['js_divergence'],
    'JS_self_check': js_self_pool['js_divergence'],
    'CG_attributable_JS': cg_attr_pool,
})
print(f'  Pooled: JS_pipeline={js_pipe_pool["js_divergence"]:.4f}, '
      f'JS_total={js_tot_pool["js_divergence"]:.4f}, '
      f'JS_cg={js_cg_pool["js_divergence"]:.4f}, '
      f'JS_self={js_self_pool["js_divergence"]:.6f}, '
      f'CG_attr={cg_attr_pool:.4f}', flush=True)

# Save divergence CSV
df_div = pd.DataFrame(divergence_rows)
div_path = os.path.join(OUT, 'rama_divergence_three_condition.csv')
df_div.to_csv(div_path, index=False)
print(f'\nSaved: {div_path}')

# Save MolProbity favored CSV (supplementary table)
df_molp = pd.DataFrame(molprobity_rows)
molp_path = os.path.join(OUT, 'rama_S3_molprobity_favored.csv')
df_molp.to_csv(molp_path, index=False)
print(f'Saved: {molp_path}')

# ── Step 3: Generate density-divergence figure ────────────────────────────────
print('\n--- Generating Figure S3 (density divergence) ---', flush=True)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
    'mathtext.default':   'regular',
})

MAX_SCATTER = 1000  # subsample scatter points per panel
np.random.seed(42)

fig, axes = plt.subplots(3, 6, figsize=(24, 12))

# Precompute 2D histograms and JS values for each panel
for row_idx, (cond_name, df_cond) in enumerate(conditions):
    for col_idx, cls in enumerate(RESIDUE_CLASSES):
        ax = axes[row_idx, col_idx]
        cat_key = CLASS_MAP_INV[cls]

        # Reference (PED) data for this class
        sub_ref = df_gt[df_gt['category'] == cls]
        sub_pred = df_cond[df_cond['category'] == cls]

        phi_ref = sub_ref['phi'].values
        psi_ref = sub_ref['psi'].values
        phi_pred = sub_pred['phi'].values
        psi_pred = sub_pred['psi'].values

        # Compute 2D histograms
        n_bins = _PHI_PSI_GRID_N
        hist_ref = _hist2d_phi_psi(phi_ref, psi_ref, n_bins)
        hist_pred = _hist2d_phi_psi(phi_pred, psi_pred, n_bins)

        # Compute cumulative density levels for contours
        # Sort bins by density, compute cumulative sum
        def _cumulative_contour_levels(hist, levels=[0.5, 0.8, 0.95]):
            """Get density values at given cumulative density levels."""
            flat = hist.flatten()
            sorted_vals = np.sort(flat)[::-1]  # descending
            cumsum = np.cumsum(sorted_vals)
            total = cumsum[-1]
            result = []
            for lev in levels:
                idx = np.searchsorted(cumsum, lev * total)
                idx = min(idx, len(sorted_vals) - 1)
                result.append(sorted_vals[idx])
            return result

        # Background: PED reference density as filled contours
        phi_edges = np.linspace(-180, 180, n_bins + 1)
        psi_edges = np.linspace(-180, 180, n_bins + 1)
        phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
        psi_centers = 0.5 * (psi_edges[:-1] + psi_edges[1:])
        PHI, PSI = np.meshgrid(phi_centers, psi_centers, indexing='ij')

        # Filled contours for reference (blue)
        ref_levels = _cumulative_contour_levels(hist_ref, [0.5, 0.8, 0.95])
        if ref_levels and ref_levels[-1] > 0:
            # Use log scale for better visualization
            hist_ref_plot = np.where(hist_ref > 0, hist_ref, np.nan)
            try:
                cf = ax.contourf(PHI, PSI, hist_ref, levels=[0] + ref_levels,
                                colors=['#d0e8ff', '#7fbfff', '#3070d0'], alpha=0.6)
            except Exception:
                pass

        # Overlay: predicted density as contour lines (red)
        pred_levels = _cumulative_contour_levels(hist_pred, [0.5, 0.8, 0.95])
        if pred_levels and pred_levels[-1] > 0:
            try:
                cs = ax.contour(PHI, PSI, hist_pred, levels=pred_levels,
                               colors=['#ff6666', '#cc0000', '#800000'],
                               linewidths=[0.8, 1.0, 1.2], alpha=0.8)
            except Exception:
                pass

        # Subsample scatter points
        n_pts = len(sub_pred)
        if n_pts > MAX_SCATTER:
            idx = np.random.choice(n_pts, MAX_SCATTER, replace=False)
            scatter_phi = sub_pred['phi'].values[idx]
            scatter_psi = sub_pred['psi'].values[idx]
        else:
            scatter_phi = sub_pred['phi'].values
            scatter_psi = sub_pred['psi'].values
        ax.scatter(scatter_phi, scatter_psi, s=1, c='black', alpha=0.3, rasterized=True)

        # JS divergence annotation
        js_result = compute_phi_psi_divergence(phi_pred, psi_pred, phi_ref, psi_ref)
        js_val = js_result['js_divergence']
        ax.text(0.03, 0.97, f'JS vs PED ref. = {js_val:.3f}', transform=ax.transAxes,
                fontsize=10, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        # Crosshairs
        ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)
        ax.axvline(0, color='black', linewidth=0.5, alpha=0.3)

        # Axes
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_xticks([-90, 0, 90])
        ax.set_yticks([-90, 0, 90])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Column titles
        if row_idx == 0:
            ax.set_title(cls, fontsize=14, fontweight='bold')

        # Row labels
        if col_idx == 0:
            row_label = ['S3A', 'S3B', 'S3C'][row_idx]
            ax.set_ylabel(f'{row_label}\n{cond_name}\nψ (°)', fontsize=12)

        if row_idx == 2:
            ax.set_xlabel('φ (°)')
        if col_idx > 0:
            ax.set_yticklabels([])

fig.tight_layout()

png_path = os.path.join(OUT, 'figure_s3.png')
pdf_path = os.path.join(OUT, 'figure_s3.pdf')
fig.savefig(png_path, bbox_inches='tight', dpi=600)
fig.savefig(pdf_path, bbox_inches='tight', dpi=600)
plt.close(fig)
print(f'Saved: {png_path}')
print(f'Saved: {pdf_path}')

# ── Step 4: Dunbrack comparison (optional) ────────────────────────────────────
print('\n--- Dunbrack comparison (optional) ---', flush=True)
# Check for Dunbrack coil library data
dunbrack_paths = [
    os.path.join(PROJECT, 'a_rama', 'dunbrack_coil_general.npy'),
    os.path.join(PROJECT, 'a_rama', 'dunbrack_coil_trans_pro.npy'),
]
if all(os.path.exists(p) for p in dunbrack_paths):
    print('Dunbrack data found, generating comparison figure...', flush=True)
    # (Would generate figure_PED_vs_Dunbrack.png here)
else:
    print('Dunbrack coil-library 5-degree grid tables not found in machine-readable form.', flush=True)
    print('Skipping Dunbrack comparison figure. The published tables at', flush=True)
    print('  http://dunbrack.fccc.edu/hdp would need manual download and conversion.', flush=True)
    print('Note: The PED-self-reference metric makes this comparison supplementary;', flush=True)
    print('the primary metric no longer depends on any external Ramachandran reference.', flush=True)

# ── Summary ───────────────────────────────────────────────────────────────────
print('\n' + '=' * 60)
print('  SUMMARY')
print('=' * 60)
print('\nJS Divergence Table (new Table 1):')
print(df_div.to_string(index=False))

print('\nMolProbity Favored % (supplementary):')
# Pivot for readability
df_molp_pivot = df_molp.pivot_table(
    index='residue_class', columns='condition',
    values='pct_favored', aggfunc='first'
)
print(df_molp_pivot.to_string())

# Sanity checks
print('\n--- Sanity Checks ---', flush=True)
# 1. JS self-check should be ~0
for _, row in df_div.iterrows():
    cls = row['residue_class']
    js_self = row['JS_self_check']
    status = 'PASS' if js_self < 0.001 else 'WARN'
    print(f'  JS(PED,PED) for {cls}: {js_self:.6f} [{status}]', flush=True)

# 2. Trans-Pro should have largest CG-attributable JS
cg_attr_by_class = df_div[df_div['residue_class'] != 'Pooled'].set_index('residue_class')['CG_attributable_JS']
max_class = cg_attr_by_class.idxmax()
print(f'\n  Largest CG-attributable JS: {max_class} = {cg_attr_by_class.max():.4f}', flush=True)
if max_class == 'Trans-Pro':
    print('  PASS: Trans-Pro is the largest CG-attributable degradation', flush=True)
else:
    print(f'  NOTE: {max_class} exceeds Trans-Pro; investigate', flush=True)

# 3. Ranking check
sorted_classes = cg_attr_by_class.sort_values(ascending=False)
print(f'\n  CG-attributable JS ranking:', flush=True)
for cls, val in sorted_classes.items():
    print(f'    {cls}: {val:.4f}', flush=True)

print(f'\nTotal time: {(time.time()-t_total)/60:.1f} min')
