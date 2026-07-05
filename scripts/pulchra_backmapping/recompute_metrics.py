"""Recompute the 7 structural metrics on PULCHRA-rebuilt caches (CG + PED conditions).

Reuses the exact metric function bodies from fig3_step100_recompute_metrics.py /
fig2_step100_worker.py so PULCHRA and CODLADAD are scored identically. Reads the
fig3-compatible .npz + _top.pkl caches produced by run_pulchra.py, writes
per_frame_metrics.csv and per_system_summary.csv (same columns as fig3).

Usage:
  python scripts/pulchra_backmapping/recompute_metrics.py --condition cg
  python scripts/pulchra_backmapping/recompute_metrics.py --condition ped
"""
import os, sys, io, pickle, time, argparse, warnings, glob
warnings.filterwarnings("ignore")

PROJECT = '/MDdata/data04/jxhuang/cg_cascade'
os.chdir(PROJECT)
sys.path.insert(0, os.path.join(PROJECT, 'src/cascade_codlad/eval_ensemble'))

import numpy as np
import pandas as pd
import mdtraj as md
from collections import defaultdict
from metrics_ramachandran import _in_boxes, _get_region_boxes

# -- Metric functions (copied verbatim from fig3_step100_recompute_metrics.py) --

def measure_rg_dmax(xyz, top):
    traj = md.Trajectory(xyz[np.newaxis, ...] / 10.0, topology=top)
    rg = float(md.compute_rg(traj)[0]) * 10.0
    ca = top.select('name CA')
    dmax = float('nan')
    if len(ca) >= 2:
        pairs = top.select_pairs(ca, ca)
        dists = md.compute_distances(traj, pairs)[0]
        dmax = float(np.max(dists)) * 10.0
    return rg, dmax

def measure_rotamer(xyz, top):
    traj = md.Trajectory(xyz[np.newaxis, ...] / 10.0, topology=top)
    _, chi1 = md.compute_chi1(traj)
    n_fav = n_alw = n_out = n_tot = 0
    for i in range(len(chi1[0])):
        c = np.degrees(chi1[0, i])
        while c > 180: c -= 360
        while c <= -180: c += 360
        d = [min(abs(c - 180), abs(c + 180)),
             min(abs(c - 60), abs(c + 300)),
             min(abs(c + 60), abs(c - 300))]
        md_ = min(d)
        n_tot += 1
        if md_ < 40: n_fav += 1
        elif md_ < 60: n_alw += 1
        else: n_out += 1
    return n_fav, n_alw, n_out, n_tot

def measure_cbdev(xyz, top):
    vals = []
    for res in top.residues:
        abn = {a.name: a.index for a in res.atoms}
        if not all(k in abn for k in ('N','CA','C','CB')): continue
        n = xyz[abn['N']]; ca = xyz[abn['CA']]; c = xyz[abn['C']]; cb = xyz[abn['CB']]
        v1 = n - ca; v2 = c - ca
        v1n = v1 / np.linalg.norm(v1); v2n = v2 / np.linalg.norm(v2)
        b3 = np.cross(v1n, v2n); cn = np.linalg.norm(b3)
        if cn < 1e-6: continue
        b3n = b3 / cn
        cos_ncac = np.dot(v1n, v2n)
        denom = 1 - cos_ncac**2
        if denom < 1e-6: continue
        a = (np.cos(np.radians(110.5)) - np.cos(np.radians(110.1)) * cos_ncac) / denom
        c_ = (np.cos(np.radians(110.1)) - np.cos(np.radians(110.5)) * cos_ncac) / denom
        b = np.sqrt(max(1 - a**2 - c_**2 - 2*a*c_*cos_ncac, 0))
        ideal = ca + 1.521 * (a * v1n + c_ * v2n + b * b3n)
        vals.append(float(np.linalg.norm(ideal - cb)))
    return vals

def measure_bond_geometry(xyz, top):
    rows = defaultdict(list)
    for res in top.residues:
        an = {a.name: a.index for a in res.atoms}
        if not all(k in an for k in ('N','CA','C')): continue
        rows['N_CA'].append(float(np.linalg.norm(xyz[an['N']] - xyz[an['CA']])))
        rows['CA_C'].append(float(np.linalg.norm(xyz[an['CA']] - xyz[an['C']])))
        if res.index + 1 < top.n_residues:
            nn = {a.name: a.index for a in top.residue(res.index + 1).atoms}
            if 'N' in nn:
                rows['C_N'].append(float(np.linalg.norm(xyz[an['C']] - xyz[nn['N']])))
    return {k: np.mean(v) for k, v in rows.items()} if rows else {}

def measure_ramachandran(xyz, top):
    traj = md.Trajectory(xyz[np.newaxis, ...] / 10.0, topology=top)
    phi_d = np.degrees(md.compute_phi(traj)[1][0])
    psi_d = np.degrees(md.compute_psi(traj)[1][0])
    omega_d = np.degrees(md.compute_omega(traj)[1][0])
    rn = [r.name for r in top.residues]
    rama_rows = []; pro_phis = []
    for i, res3 in enumerate(rn):
        nxt = rn[i+1] if i+1 < len(rn) else 'XXX'
        p = float(phi_d[i-1]) if 0 < i < len(phi_d)+1 else np.nan
        s = float(psi_d[i]) if i < len(psi_d) else np.nan
        o = float(omega_d[i-1]) if 0 < i < len(omega_d)+1 else np.nan
        if np.isnan(p) or np.isnan(s):
            rama_rows.append({'residue_idx': i, 'residue_type': res3, 'category': 'unknown', 'classification': 'unknown'})
            continue
        if res3 == 'GLY': cat = 'glycine'
        elif res3 == 'PRO': cat = 'cis_proline' if (not np.isnan(o) and abs(o) < 90) else 'trans_proline'
        elif nxt == 'PRO': cat = 'pre_proline'
        elif res3 in ('ILE', 'VAL'): cat = 'ile_val'
        else: cat = 'general'
        result = _get_region_boxes(res3, nxt)
        if result:
            fb, ab = result
            cls = 'favored' if _in_boxes(p, s, fb) else ('allowed' if _in_boxes(p, s, ab) else 'outlier')
        else: cls = 'unknown'
        rama_rows.append({'residue_idx': i, 'residue_type': res3, 'category': cat, 'classification': cls})
        if cat == 'trans_proline': pro_phis.append(p)
    return rama_rows, pro_phis

def measure_clash(xyz, top):
    """Heavy-atom vdW clash count (0.4 Å overlap threshold), 1-2/1-3 excluded.
    PULCHRA output has no H, so heavy-atom-only (matches the fig3 batch path which
    also skips PDBFixer H-addition). Uses a KDTree cutoff to avoid O(N²) all-pairs."""
    from scipy.spatial import cKDTree
    VDW_RADII = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80}
    heavy = [(a.index, a.element.symbol) for a in top.atoms
             if a.element is not None and a.element.symbol != 'H']
    if len(heavy) < 2: return 0
    heavy_idx = np.array([h[0] for h in heavy], dtype=int)
    pos2elem = [h[1] for h in heavy]
    heavy_pos = xyz[heavy_idx] / 10.0  # Å -> nm for KDTree consistency

    # 1-2 and 1-3 bonded pairs to exclude (only among heavy atoms)
    bonded = set()
    atom_neighbors = defaultdict(set)
    for bond in top.bonds:
        a1, a2 = bond[0].index, bond[1].index
        bonded.add((min(a1, a2), max(a1, a2)))
        atom_neighbors[a1].add(a2); atom_neighbors[a2].add(a1)
    for a in atom_neighbors:
        for n1 in atom_neighbors[a]:
            for n2 in atom_neighbors[a]:
                if n1 < n2: bonded.add((n1, n2))

    # Max vdW sum = 1.80+1.80 = 3.60 Å; clash if r_vdw - dist > 0.4 => dist < r_vdw - 0.4.
    # Use 3.2 Å (nm: 0.32) as the query radius (>= max possible clash distance).
    tree = cKDTree(heavy_pos)
    pairs = tree.query_pairs(r=0.32, output_type='ndarray')
    if len(pairs) == 0: return 0
    n_clash = 0
    for p in pairs:
        gi1, gi2 = int(heavy_idx[p[0]]), int(heavy_idx[p[1]])
        if (min(gi1, gi2), max(gi1, gi2)) in bonded: continue
        dist_A = float(np.linalg.norm(heavy_pos[p[0]] - heavy_pos[p[1]])) * 10.0
        r_vdw = VDW_RADII.get(pos2elem[p[0]], 1.70) + VDW_RADII.get(pos2elem[p[1]], 1.70)
        if r_vdw - dist_A > 0.4: n_clash += 1
    return n_clash

def measure_dssp(xyz, top):
    try:
        traj = md.Trajectory(xyz[np.newaxis, ...] / 10.0, topology=top)
        dssp = md.compute_dssp(traj, simplified=False)
        n_helix = n_sheet = n_coil = n_total = 0
        for ss in dssp[0]:
            n_total += 1
            if ss in ('H', 'G', 'I'): n_helix += 1
            elif ss in ('E', 'B'): n_sheet += 1
            else: n_coil += 1
        return n_helix, n_sheet, n_coil, n_total
    except Exception: return 0, 0, 0, 0

def measure_ca_fidelity(xyz_recon, top, ca_input_A):
    ca_idx = [a.index for a in top.atoms if a.name == 'CA']
    if not ca_idx: return {'max_ca_deviation': 0.0, 'mean_ca_deviation': 0.0}
    diffs = []
    for i, ci in enumerate(ca_idx):
        if ci < len(xyz_recon) and 0 <= i < len(ca_input_A):
            diffs.append(float(np.linalg.norm(xyz_recon[ci] - ca_input_A[i])))
    return {
        'max_ca_deviation': float(np.max(diffs)) if diffs else 0.0,
        'mean_ca_deviation': float(np.mean(diffs)) if diffs else 0.0,
    }

# -- Per-frame loop -------------------------------------------------------------

def recompute(condition):
    cache_dir = os.path.join(PROJECT, 'logs', f'pulchra_{condition}')
    out_dir = cache_dir
    cache_files = sorted([f for f in os.listdir(cache_dir) if f.endswith('.npz')])
    print(f'[{condition}] Found {len(cache_files)} cached frames', flush=True)

    rows = []
    t0 = time.time()
    for i, npz_file in enumerate(cache_files):
        key = npz_file[:-4]
        ped_id, frame_idx = key.rsplit('_', 1)
        frame_idx = int(frame_idx)
        npz_path = os.path.join(cache_dir, npz_file)
        top_path = os.path.join(cache_dir, f'{key}_top.pkl')
        if not os.path.exists(top_path): continue
        data = np.load(npz_path, allow_pickle=True)
        xyz_r = data['xyz_recon']
        ca_input = data['ca_input']
        with open(top_path, 'rb') as f:
            top = pickle.load(f)

        sys.stdout = io.StringIO()
        try:
            rg, dmax = measure_rg_dmax(xyz_r, top)
            rf, ra, ro, rt = measure_rotamer(xyz_r, top)
            cbs = measure_cbdev(xyz_r, top)
            bonds = measure_bond_geometry(xyz_r, top)
            rama_rows, pro_phis = measure_ramachandran(xyz_r, top)
            clash = measure_clash(xyz_r, top)
            h, s, c, nt = measure_dssp(xyz_r, top)
            cafid = measure_ca_fidelity(xyz_r, top, ca_input)
        finally:
            sys.stdout = sys.__stdout__

        rama_stats = defaultdict(lambda: {'favored':0,'allowed':0,'outlier':0,'total':0})
        for r in rama_rows:
            cl = r['classification']
            if cl == 'unknown': continue
            rama_stats[r['category']]['total'] += 1
            rama_stats[r['category']][cl] += 1

        row = {
            'ped_id': ped_id, 'frame_idx': frame_idx,
            'rg': rg, 'dmax': dmax,
            'rota_fav': rf, 'rota_alw': ra, 'rota_out': ro, 'rota_tot': rt,
            'cbdev_mean': float(np.mean(cbs)) if cbs else np.nan,
            'clash': clash,
            'dssp_helix': h, 'dssp_total': nt,
            'ca_mean_deviation': cafid['mean_ca_deviation'],
            'pro_phi_mean': float(np.mean(pro_phis)) if pro_phis else np.nan,
            'pro_phi_n': len(pro_phis),
        }
        for cat in ['general','glycine','pre_proline','trans_proline','cis_proline','ile_val']:
            row[f'rama_{cat}_fav'] = rama_stats.get(cat,{}).get('favored',0)
            row[f'rama_{cat}_tot'] = rama_stats.get(cat,{}).get('total',0)
        row['rama_pooled_fav'] = sum(c['favored'] for c in rama_stats.values())
        row['rama_pooled_tot'] = sum(c['total'] for c in rama_stats.values())
        for bt in ['CA_C','N_CA','C_N']:
            row[f'bond_{bt}'] = float(bonds.get(bt, np.nan))
        rows.append(row)
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i+1) / elapsed
            eta = (len(cache_files) - i - 1) / rate
            print(f'  [{i+1}/{len(cache_files)}] {elapsed:.0f}s, {rate:.1f}/s, ETA {eta:.0f}s', flush=True)

    df = pd.DataFrame(rows)
    pf_path = os.path.join(out_dir, 'per_frame_metrics.csv')
    df.to_csv(pf_path, index=False)
    print(f'[{condition}] Saved {len(df)} per-frame rows to {pf_path}')

    # -- Per-system summary (mirrors fig3 build_per_system_summary) -------------
    summary_rows = []
    for ped_id, sub in df.groupby('ped_id'):
        row = {'ped_id': ped_id, 'n_frames': len(sub)}
        row['rg_mean'] = sub['rg'].mean(); row['rg_std'] = sub['rg'].std()
        row['dmax_mean'] = sub['dmax'].mean()
        fav = sub['rama_pooled_fav'].sum(); tot = sub['rama_pooled_tot'].sum()
        row['rama_favored_pct'] = fav / tot * 100 if tot else np.nan
        fav = sub['rota_fav'].sum(); tot = sub['rota_tot'].sum()
        row['rota_favored_pct'] = fav / tot * 100 if tot else np.nan
        vals = sub['cbdev_mean'].dropna()
        row['cbdev_mean'] = vals.mean() if len(vals) else np.nan
        row['clash_mean'] = sub['clash'].mean()
        h = sub['dssp_helix'].sum(); t = sub['dssp_total'].sum()
        row['helix_frac'] = h / t if t else np.nan
        row['ca_mean_dev'] = sub['ca_mean_deviation'].mean()
        vals = sub['pro_phi_mean'].dropna()
        row['pro_phi_mean'] = vals.mean() if len(vals) else np.nan
        for bt in ['CA_C','N_CA','C_N']:
            row[f'bond_{bt}_mean'] = sub[f'bond_{bt}'].mean()
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    sp = os.path.join(out_dir, 'per_system_summary.csv')
    summary.to_csv(sp, index=False)
    print(f'[{condition}] Saved per-system summary ({len(summary)} systems) to {sp}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--condition', choices=['ped', 'cg', 'both'], default='both')
    args = ap.parse_args()
    conds = ['ped','cg'] if args.condition == 'both' else [args.condition]
    for c in conds:
        recompute(c)

if __name__ == '__main__':
    main()
