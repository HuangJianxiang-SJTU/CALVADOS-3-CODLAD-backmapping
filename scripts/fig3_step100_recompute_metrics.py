"""Recompute per-frame metrics from cache for all 11500 frames.
Standalone - only needs mdtraj, numpy, pandas (no torch/CODLAD)."""
import os, sys, io, pickle, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import mdtraj as md
from collections import defaultdict

PROJECT = '/MDdata/data04/jxhuang/cg_cascade'
sys.path.insert(0, os.path.join(PROJECT, 'src/cascade_codlad/eval_ensemble'))
from metrics_ramachandran import _in_boxes, _get_region_boxes

CACHE = os.path.join(PROJECT, 'logs/figure3_step100/cache')
OUT = os.path.join(PROJECT, 'logs/figure3_step100')

# -- Metric functions (copied from worker) ----------------------------------------

def add_hydrogens_pdbfixer(xyz, top, tmp_id):
    return None  # Skip hydrogen addition - too slow for batch recompute

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

def measure_ramachandran(xyz, top, tmp_id):
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

def measure_clash(xyz, top, tmp_id):
    VDW_RADII = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80}
    h_pdb = add_hydrogens_pdbfixer(xyz, top, tmp_id)
    if h_pdb is None: return 0
    try:
        t = md.load(h_pdb)
        bonded = set()
        for bond in t.topology.bonds:
            a1, a2 = bond[0].index, bond[1].index
            bonded.add((min(a1, a2), max(a1, a2)))
        atom_neighbors = defaultdict(set)
        for bond in t.topology.bonds:
            a1, a2 = bond[0].index, bond[1].index
            atom_neighbors[a1].add(a2); atom_neighbors[a2].add(a1)
        for a in range(t.n_atoms):
            for n1 in atom_neighbors[a]:
                for n2 in atom_neighbors[a]:
                    if n1 < n2: bonded.add((n1, n2))
        heavy_idx = [a.index for a in t.topology.atoms if a.element.symbol != 'H']
        heavy_pairs = t.topology.select_pairs(heavy_idx, heavy_idx)
        n_clash = 0
        if len(heavy_pairs) > 0:
            dists = md.compute_distances(t, heavy_pairs)[0] * 10.0
            for pi in range(len(heavy_pairs)):
                a1, a2 = int(heavy_pairs[pi, 0]), int(heavy_pairs[pi, 1])
                if (min(a1, a2), max(a1, a2)) in bonded: continue
                e1 = t.topology.atom(a1).element.symbol
                e2 = t.topology.atom(a2).element.symbol
                r_vdw = VDW_RADII.get(e1, 1.70) + VDW_RADII.get(e2, 1.70)
                if r_vdw - dists[pi] > 0.4: n_clash += 1
    except Exception: n_clash = 0
    if os.path.exists(h_pdb): os.remove(h_pdb)
    return n_clash

def measure_dssp(xyz, top, tmp_id):
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
        input_idx = i + 1  # CG has extra N-terminal residue
        if ci < len(xyz_recon) and 0 <= input_idx < len(ca_input_A):
            diffs.append(float(np.linalg.norm(xyz_recon[ci] - ca_input_A[input_idx])))
    return {
        'max_ca_deviation': float(np.max(diffs)) if diffs else 0.0,
        'mean_ca_deviation': float(np.mean(diffs)) if diffs else 0.0,
    }

# -- Main ------------------------------------------------------------------------

cache_files = sorted([f for f in os.listdir(CACHE) if f.endswith('.npz')])
print(f'Found {len(cache_files)} cached frames', flush=True)

rows = []
t0 = time.time()
for i, npz_file in enumerate(cache_files):
    key = npz_file[:-4]
    ped_id, frame_idx = key.rsplit('_', 1)
    frame_idx = int(frame_idx)

    npz_path = os.path.join(CACHE, npz_file)
    top_path = os.path.join(CACHE, f'{key}_top.pkl')
    if not os.path.exists(top_path): continue

    data = np.load(npz_path, allow_pickle=True)
    xyz_r = data['xyz_recon']
    ca_input = data['ca_input']
    with open(top_path, 'rb') as f:
        top = pickle.load(f)

    conf_key = key
    sys.stdout = io.StringIO()
    try:
        rg, dmax = measure_rg_dmax(xyz_r, top)
        rf, ra, ro, rt = measure_rotamer(xyz_r, top)
        cbs = measure_cbdev(xyz_r, top)
        bonds = measure_bond_geometry(xyz_r, top)
        rama_rows, pro_phis = measure_ramachandran(xyz_r, top, conf_key)
        clash = measure_clash(xyz_r, top, conf_key)
        h, s, c, nt = measure_dssp(xyz_r, top, conf_key)
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
        'cbdev_std': float(np.std(cbs)) if cbs else np.nan,
        'clash': clash,
        'dssp_helix': h, 'dssp_sheet': s, 'dssp_coil': c, 'dssp_total': nt,
        'ca_mean_deviation': cafid['mean_ca_deviation'],
        'ca_max_deviation': cafid['max_ca_deviation'],
        'pro_phi_mean': float(np.mean(pro_phis)) if pro_phis else np.nan,
        'pro_phi_std': float(np.std(pro_phis)) if pro_phis else np.nan,
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
path = os.path.join(OUT, 'per_frame_metrics.csv')
df.to_csv(path, index=False)
print(f'Saved {len(df)} rows to {path}')
