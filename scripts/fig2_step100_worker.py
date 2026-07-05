"""Figure 2 Step=100 Benchmark: CODLAD pipeline upper bound on PED reference PDBs.

Pipeline: PED PDB -> CVAE(C2) -> MPNN diffusion(steps=100) -> VAE(N6) decode.
Computes structural metrics on both GT and prediction for each conformer.
Establishes the pipeline upper bound (no CG input degradation).

Usage:
  python scripts/fig2_step100_worker.py --gpu 0 --systems PED00074
  python scripts/fig2_step100_worker.py --gpu 0  # all 23 clean_v5 systems
"""
import os, sys, csv, json, random, warnings, time, io, pickle, gc, re, glob
warnings.filterwarnings("ignore")

os.chdir('/MDdata/data04/jxhuang/cg_cascade')
sys.path.insert(0, 'src/CODLAD'); sys.path.insert(0, 'src')

import numpy as np
import torch
import mdtraj as md
import pandas as pd
from collections import defaultdict

from utils.utils_ic import ic_to_xyz, core_atoms as CODLAD_CORE_ATOMS
from cascade_codlad.eval_ensemble.metrics_ramachandran import _get_region_boxes, _in_boxes

# -- Configuration ---------------------------------------------------------------
PROJECT = '/MDdata/data04/jxhuang/cg_cascade'
CODLAD_DIR = os.path.join(PROJECT, 'src/CODLAD')
OUT = os.path.join(PROJECT, 'logs/figure2_step100')
CACHE = os.path.join(OUT, 'cache')
TMP = os.path.join(PROJECT, 'data/fig2_step100_tmp')
SEED = 42
STEPS = 100      # diffusion steps (author default, NOT step=5)
N_GPUS = 4
CHUNK_SIZE = 200
MAX_CONF = 300   # cap conformers per system (reduced from 500 to fit 24h compute budget)

os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)
os.makedirs(TMP, exist_ok=True)

COD_IDX2RES = {
    0:'ASN',1:'HIS',2:'ALA',3:'GLY',4:'ARG',5:'MET',6:'SER',7:'ILE',
    8:'GLU',9:'LEU',10:'TYR',11:'ASP',12:'VAL',13:'TRP',14:'GLN',
    15:'LYS',16:'PRO',17:'PHE',18:'CYS',19:'THR',
}

NONSTANDARD_RES = {'RCY', 'ZN', 'FE', 'CU', 'MG', 'MN', 'CA', 'CL', 'NA', 'K', 'HOH', 'SO4'}

# -- PED discovery ---------------------------------------------------------------

def discover_peds():
    sources = {
        'ped_ensembles': 'data/processed/cascade/ped_ensembles',
        'ped_staged':    'data/processed/cascade/ped_staged',
        'ped_raw':       'data/processed/cascade/ped_raw',
    }
    peds = {}
    priority = {'ped_ensembles': 1, 'ped_staged': 2, 'ped_raw': 3}
    for src_name, src_dir in sources.items():
        if not os.path.isdir(src_dir): continue
        if src_name == 'ped_ensembles':
            for ped_id in os.listdir(src_dir):
                ped_path = os.path.join(src_dir, ped_id)
                if not os.path.isdir(ped_path): continue
                pdbs = sorted([f for f in os.listdir(ped_path) if f.endswith('.pdb')])
                if pdbs:
                    pdb_path = os.path.join(ped_path, pdbs[0])
                    if ped_id not in peds or priority[src_name] < peds[ped_id][2]:
                        peds[ped_id] = (src_name, pdb_path, priority[src_name])
        else:
            for pdb_file in glob.glob(os.path.join(src_dir, '*.pdb')):
                ped_id_match = re.search(r'PED\d+', os.path.basename(pdb_file))
                if ped_id_match:
                    ped_id = ped_id_match.group()
                    if ped_id not in peds or priority[src_name] < peds[ped_id][2]:
                        peds[ped_id] = (src_name, pdb_file, priority[src_name])
    return peds

# -- Lightweight PDB tools -------------------------------------------------------

def quick_pdb_check(pdb_path):
    n_models = 0
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("MODEL"): n_models += 1
    chains = set(); residues = set(); has_nonstandard = False
    in_first_model = True; seen_endmdl = False
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("MODEL"):
                if seen_endmdl: break
                continue
            elif line.startswith("ENDMDL"):
                seen_endmdl = True; continue
            elif line.startswith(("ATOM","HETATM")):
                try:
                    resname = line[17:20].strip()
                    atomname = line[12:16].strip()
                    if resname in NONSTANDARD_RES: has_nonstandard = True
                    if atomname[0] != 'H':
                        chains.add(line[21])
                        residues.add((line[21], line[22:26].strip(), line[26].strip() if len(line)>26 else ''))
                except: continue
    if n_models == 0: n_models = 1
    return n_models, len(chains), has_nonstandard, len(residues)

def read_sampled_models(pdb_path, model_indices):
    needed = set(model_indices)
    models = {}; current_model = None; current_lines = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("MODEL"):
                current_model = int(line.split()[1]) - 1
                current_lines = [line] if current_model in needed else []
            elif line.startswith("ENDMDL"):
                if current_model is not None and current_model in needed:
                    current_lines.append("END\n")
                    models[current_model] = current_lines
                current_model = None; current_lines = []
            elif current_model is not None and current_model in needed:
                if line.startswith(("ATOM","HETATM")) and line[12:16].strip()[:1] == 'H': continue
                current_lines.append(line)
    return models

def reorder_to_codlad(lines, core_atoms):
    residues = {}; res_order = []; model_line = None; end_line = None
    for line in lines:
        if line.startswith("MODEL"): model_line = line; continue
        if line.startswith("END"): end_line = line; continue
        if line.startswith(("ATOM", "HETATM")):
            key = (line[21], line[17:20].strip(), line[22:26])
            if key not in residues: residues[key] = {}; res_order.append(key)
            residues[key][line[12:16].strip()] = line
    out = [model_line] if model_line else []
    for key in res_order:
        _, resname, _ = key
        codlad_atoms = core_atoms.get(resname, core_atoms.get('ALA', ['O','N','C','CA']))
        ca_line = residues[key].get('CA', None)
        for aname in codlad_atoms:
            if aname in residues[key]:
                out.append(residues[key][aname])
            elif ca_line:
                p = list(ca_line)
                nm = f' {aname:<3s}' if len(aname) < 4 else f'{aname:<4s}'
                out.append(''.join(p[:12] + list(nm) + p[16:]))
    if end_line: out.append(end_line)
    return out

# -- Metric helpers (with all bug fixes applied) ---------------------------------

def measure_bond_geometry(xyz, top):
    rows = []
    for residue in top.residues:
        ri = residue.index; atoms_by_name = {a.name: a.index for a in residue.atoms}
        if 'CA' not in atoms_by_name or 'N' not in atoms_by_name or 'C' not in atoms_by_name: continue
        n_pos = xyz[atoms_by_name['N']]; ca_pos = xyz[atoms_by_name['CA']]; c_pos = xyz[atoms_by_name['C']]
        rows.append({'residue_idx': ri, 'residue_type': residue.name, 'bond_type': 'N_CA',
                      'length': float(np.linalg.norm(n_pos - ca_pos))})
        rows.append({'residue_idx': ri, 'residue_type': residue.name, 'bond_type': 'CA_C',
                      'length': float(np.linalg.norm(ca_pos - c_pos))})
        if ri + 1 < top.n_residues:
            next_atoms = {a.name: a.index for a in top.residue(ri + 1).atoms}
            if 'N' in next_atoms:
                rows.append({'residue_idx': ri, 'residue_type': residue.name, 'bond_type': 'C_N',
                              'length': float(np.linalg.norm(c_pos - xyz[next_atoms['N']]))})
    return rows

def add_hydrogens_pdbfixer(xyz, top, tmp_id):
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile
    heavy_pdb = os.path.join(TMP, f'{tmp_id}_heavy.pdb')
    h_pdb = os.path.join(TMP, f'{tmp_id}_h.pdb')
    traj_heavy = md.Trajectory(xyz[np.newaxis, ...] / 10.0, topology=top)
    traj_heavy.save(heavy_pdb)
    try:
        fixer = PDBFixer(filename=heavy_pdb)
        fixer.addMissingHydrogens(pH=7.0)
        with open(h_pdb, 'w') as f: PDBFile.writeFile(fixer.topology, fixer.positions, f)
    except:
        for p in [heavy_pdb, h_pdb]:
            if os.path.exists(p): os.remove(p)
        return None
    if os.path.exists(heavy_pdb): os.remove(heavy_pdb)
    return h_pdb

def measure_ramachandran(xyz, top, tmp_id):
    """Returns (rama_rows, pro_phis, n_clash). Uses MolProbity-style clash detection."""
    traj = md.Trajectory(xyz[np.newaxis, ...] / 10.0, topology=top)
    phi_d = np.degrees(md.compute_phi(traj)[1][0])
    psi_d = np.degrees(md.compute_psi(traj)[1][0])
    omega_d = np.degrees(md.compute_omega(traj)[1][0])
    rn = [r.name for r in top.residues]
    rama_rows = []; pro_phis = []
    for i, res3 in enumerate(rn):
        nxt = rn[i+1] if i+1 < len(rn) else 'XXX'
        # Fix: phi[i-1] and omega[i-1] give the dihedral BEFORE residue i
        p = float(phi_d[i-1]) if 0 < i < len(phi_d)+1 else np.nan
        s = float(psi_d[i]) if i < len(psi_d) else np.nan
        o = float(omega_d[i-1]) if 0 < i < len(omega_d)+1 else np.nan
        if np.isnan(p) or np.isnan(s):
            rama_rows.append({'residue_idx': i, 'residue_type': res3, 'phi': p, 'psi': s,
                              'category': 'unknown', 'classification': 'unknown'}); continue
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
        rama_rows.append({'residue_idx': i, 'residue_type': res3, 'phi': p, 'psi': s,
                          'category': cat, 'classification': cls})
        if cat == 'trans_proline': pro_phis.append(p)

    # Fix: MolProbity-style clash detection (vdW overlap > 0.4 A, excluding 1-2 and 1-3 bonded pairs)
    VDW_RADII = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80}
    n_clash = 0
    h_pdb = add_hydrogens_pdbfixer(xyz, top, tmp_id)
    if h_pdb is not None:
        try:
            t = md.load(h_pdb)
            # Build bonded pair exclusion set (1-2 and 1-3)
            bonded = set()
            for bond in t.topology.bonds:
                a1, a2 = bond[0].index, bond[1].index
                bonded.add((min(a1, a2), max(a1, a2)))
            atom_neighbors = defaultdict(set)
            for bond in t.topology.bonds:
                a1, a2 = bond[0].index, bond[1].index
                atom_neighbors[a1].add(a2)
                atom_neighbors[a2].add(a1)
            for a in range(t.n_atoms):
                for n1 in atom_neighbors[a]:
                    for n2 in atom_neighbors[a]:
                        if n1 < n2:
                            bonded.add((n1, n2))
            # Check heavy-atom pairs for vdW overlap
            heavy_idx = [a.index for a in t.topology.atoms if a.element.symbol != 'H']
            heavy_pairs = t.topology.select_pairs(heavy_idx, heavy_idx)
            if len(heavy_pairs) > 0:
                dists = md.compute_distances(t, heavy_pairs)[0] * 10.0  # nm -> Angstroms
                for pi in range(len(heavy_pairs)):
                    a1, a2 = int(heavy_pairs[pi, 0]), int(heavy_pairs[pi, 1])
                    if (min(a1, a2), max(a1, a2)) in bonded:
                        continue
                    e1 = t.topology.atom(a1).element.symbol
                    e2 = t.topology.atom(a2).element.symbol
                    r_vdw = VDW_RADII.get(e1, 1.70) + VDW_RADII.get(e2, 1.70)
                    overlap = r_vdw - dists[pi]
                    if overlap > 0.4:
                        n_clash += 1
        except: pass
        if os.path.exists(h_pdb): os.remove(h_pdb)
    return rama_rows, pro_phis, n_clash

def measure_rotamer(xyz_recon, top):
    traj_tmp = md.Trajectory(xyz_recon[np.newaxis, ...] / 10.0, topology=top)
    chi1_atoms, chi1_angles = md.compute_chi1(traj_tmp)
    n_fav = 0; n_alw = 0; n_out = 0; n_tot = 0
    for i in range(len(chi1_angles[0])):
        chi1 = np.degrees(chi1_angles[0, i])
        while chi1 > 180: chi1 -= 360
        while chi1 <= -180: chi1 += 360
        d1 = min(abs(chi1-180), abs(chi1+180))
        d2 = min(abs(chi1-60), abs(chi1+300))
        d3 = min(abs(chi1+60), abs(chi1-300))
        min_d = min(d1, d2, d3)
        n_tot += 1
        if min_d < 40: n_fav += 1
        elif min_d < 60: n_alw += 1
        else: n_out += 1
    return n_fav, n_alw, n_out, n_tot

def measure_rg_dmax(xyz, top):
    traj_tmp = md.Trajectory(xyz[np.newaxis, ...] / 10.0, topology=top)
    # Fix: compute_rg returns nm; convert to Angstroms
    rg = float(md.compute_rg(traj_tmp)[0]) * 10.0
    ca_idx = top.select('name CA')
    dmax = float('nan')
    if len(ca_idx) >= 2:
        pairs = top.select_pairs(ca_idx, ca_idx)
        dists = md.compute_distances(traj_tmp, pairs)[0]
        # Fix: compute_distances returns nm; convert to Angstroms
        dmax = float(np.max(dists)) * 10.0
    return rg, dmax

def measure_cbdev(xyz, top):
    vals = []
    for res in top.residues:
        abn = {a.name: a.index for a in res.atoms}
        if 'N' not in abn or 'CA' not in abn or 'C' not in abn or 'CB' not in abn: continue
        n_pos = xyz[abn['N']]; ca_pos = xyz[abn['CA']]; c_pos = xyz[abn['C']]; cb_pos = xyz[abn['CB']]
        v1 = n_pos - ca_pos; v2 = c_pos - ca_pos
        v1n = v1 / np.linalg.norm(v1); v2n = v2 / np.linalg.norm(v2)
        b3 = np.cross(v1n, v2n); cn = np.linalg.norm(b3)
        if cn < 1e-6: continue
        b3n = b3 / cn
        cos_ncac = np.dot(v1n, v2n)
        cos_t1 = np.cos(np.radians(110.5)); cos_t2 = np.cos(np.radians(110.1))
        denom = 1 - cos_ncac**2
        if denom < 1e-6: continue
        a = (cos_t1 - cos_t2 * cos_ncac) / denom
        c_coeff = (cos_t2 - cos_t1 * cos_ncac) / denom
        val = 1 - a**2 - c_coeff**2 - 2*a*c_coeff*cos_ncac
        b_coeff = np.sqrt(max(val, 0))
        cb_ideal_dir = a * v1n + c_coeff * v2n + b_coeff * b3n
        cb_ideal_pos = ca_pos + 1.521 * cb_ideal_dir
        vals.append(float(np.linalg.norm(cb_ideal_pos - cb_pos)))
    return vals

def measure_dssp(xyz, top, tmp_id):
    n_helix = 0; n_sheet = 0; n_coil = 0; n_total = 0
    try:
        traj = md.Trajectory(xyz[np.newaxis, ...] / 10.0, topology=top)
        dssp = md.compute_dssp(traj, simplified=False)
        for ss in dssp[0]:
            n_total += 1
            if ss in ('H', 'G', 'I'):
                n_helix += 1
            elif ss in ('E', 'B'):
                n_sheet += 1
            else:
                n_coil += 1
    except: pass
    return n_helix, n_sheet, n_coil, n_total

# -- Main worker -----------------------------------------------------------------

def process_systems(ped_ids, gpu_id, seed_offset=0, max_conf=None):
    """Process assigned systems on given GPU. Returns list of per-conformer results."""
    DEVICE = 'cuda:0'
    DATA_PARAMS = {"atom_cutoff": 9.0, "cg_cutoff": 21.0, "edgeorder": 2}

    random.seed(SEED + gpu_id + seed_offset)
    np.random.seed(SEED + gpu_id + seed_offset)
    torch.manual_seed(SEED + gpu_id + seed_offset)

    # -- Load models --
    _cwd = os.getcwd(); os.chdir(CODLAD_DIR)
    from utils.model_module import get_vae_model
    vae_model, _ = get_vae_model("N6", device=DEVICE, modelnum=-1); vae_model.eval()
    cvae_model, _ = get_vae_model("C2", device=DEVICE, modelnum=-1); cvae_model.eval()

    from models.latent_model import MPNN_models
    diff_model = MPNN_models['mpnn_diffusion'](
        input_size=3, unconditional=False, diffusion='diffusion', self_condition=False).to(DEVICE)
    ckpt = torch.load('results/Diff_PED_mpnnnew/protein_weights_best.pt', map_location='cpu')
    try: diff_model.load_state_dict(ckpt['net_model'])
    except:
        from collections import OrderedDict
        ns = OrderedDict()
        for k, v in ckpt['net_model'].items(): ns[k[7:]] = v
        diff_model.load_state_dict(ns)
    diff_model.eval()

    from diffusion_and_flow import create_diffusion
    # Step=100: author default diffusion step count
    diffusion = create_diffusion(str(STEPS), noise_schedule='linear', predict_xstart=False,
                                  rescale_learned_sigmas=False, self_condition=False)
    from utils.dataset_module import load_dataset, get_norm_feature
    os.chdir(_cwd)

    log_prefix = f"[GPU{gpu_id}]"

    results = []

    for pi, ped_id in enumerate(ped_ids):
        pdb_path = ALL_PEDS[ped_id][1] if ped_id in ALL_PEDS else None
        if pdb_path is None:
            print(f"{log_prefix} {ped_id}: no PDB, skip")
            continue
        print(f"{log_prefix} [{pi+1}/{len(ped_ids)}] {ped_id}: starting", flush=True)

        # Check PDB
        try:
            n_models, n_chains, has_nonstandard, n_res = quick_pdb_check(pdb_path)
        except Exception as e:
            print(f"{log_prefix} {ped_id}: parse fail ({type(e).__name__})")
            continue

        if n_chains > 1 or has_nonstandard or n_res == 0:
            print(f"{log_prefix} {ped_id}: skip (chains={n_chains}, nonstd={has_nonstandard}, res={n_res})")
            continue

        # Conformer sampling: cap at MAX_CONF with evenly-spaced striding
        _max_conf = max_conf if max_conf is not None else MAX_CONF
        if n_models > _max_conf:
            model_indices = list(np.linspace(0, n_models - 1, _max_conf, dtype=int))
        else:
            model_indices = list(range(n_models))
        sampled_models = read_sampled_models(pdb_path, model_indices)
        print(f"{log_prefix} {ped_id}: {n_models} models, sampling {len(model_indices)}", flush=True)

        # Build list of valid conformers with pre-reordered lines
        valid_indices = []
        valid_lines = {}
        for mi in model_indices:
            lines = sampled_models.get(mi, [])
            if not lines: continue
            reordered = reorder_to_codlad(lines, CODLAD_CORE_ATOMS)
            valid_indices.append(mi)
            valid_lines[mi] = reordered

        if not valid_indices:
            print(f"{log_prefix} {ped_id}: no valid conformers")
            continue

        # Count how many are already cached
        n_cached = sum(1 for mi in valid_indices if os.path.exists(os.path.join(CACHE, f'{ped_id}_{mi}.npz')))
        if n_cached == len(valid_indices):
            print(f"{log_prefix} {ped_id}: all {len(valid_indices)} conformers cached, computing metrics from cache", flush=True)
        print(f"{log_prefix} {ped_id}: {len(valid_indices)} conformers ({n_cached} cached), chunk_size={CHUNK_SIZE}", flush=True)

        # Process in chunks to avoid O(N) IC generation bottleneck
        for chunk_start in range(0, len(valid_indices), CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, len(valid_indices))
            chunk_indices = valid_indices[chunk_start:chunk_end]

            # Don't skip chunk even if all cached - we need to compute metrics
            # (the per-conformer loop will load from cache)

            # Write chunk to temp PDB
            all_pdb = os.path.join(TMP, f'g{gpu_id}_{ped_id}_chunk{chunk_start}.pdb')
            with open(all_pdb, 'w') as f:
                for mi in chunk_indices:
                    f.writelines(valid_lines[mi])

            all_stem = all_pdb[:-4]
            try:
                t0 = time.time()
                loader, info_dict, n_atoms, n_cgs, _, _top = load_dataset(all_stem, DATA_PARAMS, single=True)
                dt = time.time() - t0
                print(f"{log_prefix} {ped_id} chunk {chunk_start}-{chunk_end}: load_dataset {dt:.1f}s", flush=True)
            except Exception as e:
                print(f"{log_prefix} {ped_id} chunk {chunk_start}: load_dataset failed: {type(e).__name__}: {str(e)[:200]}")
                if os.path.exists(all_pdb): os.remove(all_pdb)
                continue

            from torch.utils.data import DataLoader as _DL
            from utils.dataset_module import CG_collate
            single_loader = _DL(loader.dataset, batch_size=1, collate_fn=CG_collate, shuffle=False, pin_memory=True)

            for batch_idx, batch in enumerate(single_loader):
                if batch_idx >= len(chunk_indices): break
                mi = chunk_indices[batch_idx]

                cache_key = f'{ped_id}_{mi}'
                cache_npz = os.path.join(CACHE, f'{cache_key}.npz')
                cache_top = os.path.join(CACHE, f'{cache_key}_top.pkl')

                # If cached, load prediction and compute metrics (don't skip)
                if os.path.exists(cache_npz) and os.path.exists(cache_top):
                    try:
                        data = np.load(cache_npz, allow_pickle=True)
                        pred_np = data['xyz_pred']
                        gt_np = data['xyz_gt']
                        interior_names = list(data['rn'])
                        with open(cache_top, 'rb') as _pf:
                            top = pickle.load(_pf)
                        conf_key = f"{ped_id}_{mi}"
                        sys.stdout = io.StringIO()
                        try:
                            rg_gt, dmax_gt = measure_rg_dmax(gt_np, top)
                            rg_pred, dmax_pred = measure_rg_dmax(pred_np, top)
                            n_rf, n_ra, n_ro, n_rt = measure_rotamer(pred_np, top)
                            n_rf_gt, n_ra_gt, n_ro_gt, n_rt_gt = measure_rotamer(gt_np, top)
                            cb_pred = measure_cbdev(pred_np, top)
                            cb_gt = measure_cbdev(gt_np, top)
                            h_gt, s_gt, c_gt, n_gt = measure_dssp(gt_np, top, f'{conf_key}_gt')
                            h_pr, s_pr, c_pr, n_pr = measure_dssp(pred_np, top, f'{conf_key}_pr')
                            rama_gt, pro_gt, clash_gt = measure_ramachandran(gt_np, top, f'{conf_key}_gt')
                            rama_pr, pro_pr, clash_pr = measure_ramachandran(pred_np, top, f'{conf_key}_pr')
                            bonds_gt = measure_bond_geometry(gt_np, top)
                            bonds_pr = measure_bond_geometry(pred_np, top)
                        finally:
                            sys.stdout = sys.__stdout__

                        # Aggregate (same code as below)
                        sq = np.sum((pred_np - gt_np)**2, axis=1)
                        bb_err = []; sc_err = []; off = 0
                        sc_err_by_type = defaultdict(list)
                        for res3 in interior_names:
                            ats = CODLAD_CORE_ATOMS.get(res3, CODLAD_CORE_ATOMS['ALA'])
                            n = len(ats)
                            if off + n > len(sq): break
                            bb_err.extend(sq[off:off+min(4,n)].tolist())
                            if n > 4:
                                sc_sq = sq[off+4:off+n].tolist()
                                sc_err.extend(sc_sq)
                                sc_err_by_type[res3].extend(sc_sq)
                            off += n
                        bb_rmsd = float(np.sqrt(np.mean(bb_err))) if bb_err else float('nan')
                        sc_rmsd = float(np.sqrt(np.mean(sc_err))) if sc_err else float('nan')
                        aa_rmsd = float(np.sqrt(np.mean(bb_err + sc_err)))
                        sc_bb_ratio = sc_rmsd / bb_rmsd if bb_rmsd > 1e-6 else float('nan')
                        sc_rmsd_by_type = {}
                        for res3 in sorted(sc_err_by_type.keys()):
                            vals = sc_err_by_type[res3]
                            sc_rmsd_by_type[res3] = float(np.sqrt(np.mean(vals))) if vals else float('nan')

                        def _rama_stats(rows):
                            stats = defaultdict(lambda: {'favored':0,'allowed':0,'outlier':0,'unknown':0,'total':0})
                            for r in rows:
                                cl = r['classification']
                                if cl == 'unknown': continue
                                stats[r['category']]['total'] += 1
                                stats[r['category']][cl] += 1
                            return stats
                        rama_gt_stats = _rama_stats(rama_gt)
                        rama_pr_stats = _rama_stats(rama_pr)
                        def _bond_stats(rows):
                            out = {}
                            for bt in ['CA_C','N_CA','C_N']:
                                vals = [r['length'] for r in rows if r['bond_type']==bt]
                                out[bt] = (np.mean(vals), np.std(vals)) if vals else (float('nan'), float('nan'))
                            return out

                        result = {
                            'ped_id': ped_id, 'conformer_idx': mi,
                            'rg_gt': rg_gt, 'rg_pred': rg_pred,
                            'dmax_gt': dmax_gt, 'dmax_pred': dmax_pred,
                            'rota_fav_gt': n_rf_gt, 'rota_alw_gt': n_ra_gt, 'rota_out_gt': n_ro_gt, 'rota_tot_gt': n_rt_gt,
                            'rota_fav_pred': n_rf, 'rota_alw_pred': n_ra, 'rota_out_pred': n_ro, 'rota_tot_pred': n_rt,
                            'cbdev_mean_gt': float(np.mean(cb_gt)) if cb_gt else float('nan'),
                            'cbdev_std_gt': float(np.std(cb_gt)) if cb_gt else float('nan'),
                            'cbdev_mean_pred': float(np.mean(cb_pred)) if cb_pred else float('nan'),
                            'cbdev_std_pred': float(np.std(cb_pred)) if cb_pred else float('nan'),
                            'clash_gt': clash_gt, 'clash_pred': clash_pr,
                            'dssp_helix_gt': h_gt, 'dssp_sheet_gt': s_gt, 'dssp_coil_gt': c_gt, 'dssp_total_gt': n_gt,
                            'dssp_helix_pred': h_pr, 'dssp_sheet_pred': s_pr, 'dssp_coil_pred': c_pr, 'dssp_total_pred': n_pr,
                            'bb_rmsd': bb_rmsd, 'sc_rmsd': sc_rmsd, 'aa_rmsd': aa_rmsd,
                            'sc_bb_ratio': sc_bb_ratio,
                        }
                        for cat in ['general','glycine','pre_proline','trans_proline','cis_proline','ile_val']:
                            result[f'rama_gt_{cat}_fav'] = rama_gt_stats.get(cat,{}).get('favored',0)
                            result[f'rama_gt_{cat}_tot'] = rama_gt_stats.get(cat,{}).get('total',0)
                            result[f'rama_pred_{cat}_fav'] = rama_pr_stats.get(cat,{}).get('favored',0)
                            result[f'rama_pred_{cat}_tot'] = rama_pr_stats.get(cat,{}).get('total',0)
                        result['rama_gt_pooled_fav'] = sum(c['favored'] for c in rama_gt_stats.values())
                        result['rama_gt_pooled_tot'] = sum(c['total'] for c in rama_gt_stats.values())
                        result['rama_pred_pooled_fav'] = sum(c['favored'] for c in rama_pr_stats.values())
                        result['rama_pred_pooled_tot'] = sum(c['total'] for c in rama_pr_stats.values())
                        result['pro_phi_mean_gt'] = float(np.mean(pro_gt)) if pro_gt else float('nan')
                        result['pro_phi_mean_pred'] = float(np.mean(pro_pr)) if pro_pr else float('nan')
                        result['pro_phi_n_gt'] = len(pro_gt)
                        result['pro_phi_n_pred'] = len(pro_pr)
                        bgt = _bond_stats(bonds_gt); bpr = _bond_stats(bonds_pr)
                        for bt in ['CA_C','N_CA','C_N']:
                            result[f'bond_gt_mean_{bt}'] = float(bgt[bt][0]); result[f'bond_gt_std_{bt}'] = float(bgt[bt][1])
                            result[f'bond_pred_mean_{bt}'] = float(bpr[bt][0]); result[f'bond_pred_std_{bt}'] = float(bpr[bt][1])
                        for res3 in sorted(sc_rmsd_by_type.keys()):
                            result[f'sc_rmsd_{res3}'] = sc_rmsd_by_type[res3]
                        results.append(result)
                    except Exception as e:
                        sys.stdout = sys.__stdout__
                        print(f"{log_prefix} {ped_id} cached conf {mi}: metric error: {type(e).__name__}: {str(e)[:200]}")
                    continue

                try:
                    with torch.no_grad():
                        from utils.train_module import batch_to as _bt
                        batch = _bt(batch, DEVICE)
                        nres = int(batch["num_CGs"][0].item()) + 2
                        OG = batch["OG_CG_nxyz"].reshape(-1, nres, 4)
                        mask_xyz = batch["mask_xyz_list"]
                        xyz_true = batch["nxyz"][:, 1:].clone(); xyz_true[mask_xyz] = 0.0

                        # -- Author pipeline: CVAE -> diffusion(100 steps) -> VAE decode --
                        y, _, _, mask, num_CGs, _, _ = cvae_model.get_latent_cg(batch)
                        seq_len = batch['num_CGs'][0].item()
                        z = torch.randn((1, seq_len, 3), device=DEVICE)
                        cat_z = torch.cat([z, z], 0)
                        y_null = torch.zeros_like(y); cat_y = torch.cat([y, y_null], 0)
                        cat_mask = torch.cat([mask, mask], 0)
                        batch["randn"] = torch.randn([cat_z.shape[0], cat_z.shape[1]], device=DEVICE)

                        samples = diffusion.p_sample_loop(
                            diff_model.forward, cat_z.shape, cat_z, clip_denoised=False,
                            model_kwargs=dict(y=cat_y, mask=cat_mask, batch=batch),
                            progress=False, device=DEVICE)
                        samples, _ = samples.chunk(2, dim=0)

                        _cwd2 = os.getcwd(); os.chdir(CODLAD_DIR)
                        samples = get_norm_feature(samples, "N6", norm_channel=True, norm_single=False,
                                                    norm_in=False, dataname="PED")
                        os.chdir(_cwd2)

                        _, ic_recon = vae_model.latent_decode(samples, mask, batch)
                        xyz_recon = ic_to_xyz(OG, ic_recon.reshape(-1, nres-2, 13, 3), info_dict[0]).reshape(-1, 3)

                        # Free intermediate tensors
                        del y, z, cat_z, y_null, cat_y, cat_mask, samples, ic_recon
                        if nres > 300:
                            torch.cuda.empty_cache()
                            gc.collect()

                        gt_np = xyz_true.cpu().numpy()
                        pred_np = xyz_recon.cpu().numpy()
                        top = _top

                        # -- Compute metrics on BOTH GT and prediction --
                        conf_key = f"{ped_id}_{mi}"
                        # Fix: use try/finally to ensure stdout is always restored
                        sys.stdout = io.StringIO()
                        try:
                            rg_gt, dmax_gt = measure_rg_dmax(gt_np, top)
                            rg_pred, dmax_pred = measure_rg_dmax(pred_np, top)

                            n_rf, n_ra, n_ro, n_rt = measure_rotamer(pred_np, top)
                            n_rf_gt, n_ra_gt, n_ro_gt, n_rt_gt = measure_rotamer(gt_np, top)

                            cb_pred = measure_cbdev(pred_np, top)
                            cb_gt = measure_cbdev(gt_np, top)

                            h_gt, s_gt, c_gt, n_gt = measure_dssp(gt_np, top, f'{conf_key}_gt')
                            h_pr, s_pr, c_pr, n_pr = measure_dssp(pred_np, top, f'{conf_key}_pr')

                            rama_gt, pro_gt, clash_gt = measure_ramachandran(gt_np, top, f'{conf_key}_gt')
                            rama_pr, pro_pr, clash_pr = measure_ramachandran(pred_np, top, f'{conf_key}_pr')

                            bonds_gt = measure_bond_geometry(gt_np, top)
                            bonds_pr = measure_bond_geometry(pred_np, top)
                        finally:
                            sys.stdout = sys.__stdout__

                        # -- Aggregate --
                        cg_types = batch["CG_nxyz"][:, 0].long().cpu().numpy()
                        interior_names = [COD_IDX2RES.get(int(t), 'ALA') for t in cg_types]

                        sq = np.sum((pred_np - gt_np)**2, axis=1)
                        bb_err = []; sc_err = []; off = 0
                        # Per-residue-type SC RMSD
                        sc_err_by_type = defaultdict(list)
                        for res3 in interior_names:
                            ats = CODLAD_CORE_ATOMS.get(res3, CODLAD_CORE_ATOMS['ALA'])
                            n = len(ats)
                            if off + n > len(sq): break
                            bb_err.extend(sq[off:off+min(4,n)].tolist())
                            if n > 4:
                                sc_sq = sq[off+4:off+n].tolist()
                                sc_err.extend(sc_sq)
                                sc_err_by_type[res3].extend(sc_sq)
                            off += n
                        bb_rmsd = float(np.sqrt(np.mean(bb_err))) if bb_err else float('nan')
                        sc_rmsd = float(np.sqrt(np.mean(sc_err))) if sc_err else float('nan')
                        aa_rmsd = float(np.sqrt(np.mean(bb_err + sc_err)))
                        sc_bb_ratio = sc_rmsd / bb_rmsd if bb_rmsd > 1e-6 else float('nan')

                        # Per-residue-type SC RMSD
                        sc_rmsd_by_type = {}
                        for res3 in sorted(sc_err_by_type.keys()):
                            vals = sc_err_by_type[res3]
                            sc_rmsd_by_type[res3] = float(np.sqrt(np.mean(vals))) if vals else float('nan')

                        def _rama_stats(rows):
                            stats = defaultdict(lambda: {'favored':0,'allowed':0,'outlier':0,'unknown':0,'total':0})
                            for r in rows:
                                cl = r['classification']
                                if cl == 'unknown': continue
                                stats[r['category']]['total'] += 1
                                stats[r['category']][cl] += 1
                            return stats

                        rama_gt_stats = _rama_stats(rama_gt)
                        rama_pr_stats = _rama_stats(rama_pr)

                        def _bond_stats(rows):
                            out = {}
                            for bt in ['CA_C','N_CA','C_N']:
                                vals = [r['length'] for r in rows if r['bond_type']==bt]
                                out[bt] = (np.mean(vals), np.std(vals)) if vals else (float('nan'), float('nan'))
                            return out

                        result = {
                            'ped_id': ped_id, 'conformer_idx': mi,
                            'rg_gt': rg_gt, 'rg_pred': rg_pred,
                            'dmax_gt': dmax_gt, 'dmax_pred': dmax_pred,
                            'rota_fav_gt': n_rf_gt, 'rota_alw_gt': n_ra_gt, 'rota_out_gt': n_ro_gt, 'rota_tot_gt': n_rt_gt,
                            'rota_fav_pred': n_rf, 'rota_alw_pred': n_ra, 'rota_out_pred': n_ro, 'rota_tot_pred': n_rt,
                            'cbdev_mean_gt': float(np.mean(cb_gt)) if cb_gt else float('nan'),
                            'cbdev_std_gt': float(np.std(cb_gt)) if cb_gt else float('nan'),
                            'cbdev_mean_pred': float(np.mean(cb_pred)) if cb_pred else float('nan'),
                            'cbdev_std_pred': float(np.std(cb_pred)) if cb_pred else float('nan'),
                            'clash_gt': clash_gt, 'clash_pred': clash_pr,
                            'dssp_helix_gt': h_gt, 'dssp_sheet_gt': s_gt, 'dssp_coil_gt': c_gt, 'dssp_total_gt': n_gt,
                            'dssp_helix_pred': h_pr, 'dssp_sheet_pred': s_pr, 'dssp_coil_pred': c_pr, 'dssp_total_pred': n_pr,
                            'bb_rmsd': bb_rmsd, 'sc_rmsd': sc_rmsd, 'aa_rmsd': aa_rmsd,
                            'sc_bb_ratio': sc_bb_ratio,
                        }
                        for cat in ['general','glycine','pre_proline','trans_proline','cis_proline','ile_val']:
                            result[f'rama_gt_{cat}_fav'] = rama_gt_stats.get(cat,{}).get('favored',0)
                            result[f'rama_gt_{cat}_tot'] = rama_gt_stats.get(cat,{}).get('total',0)
                            result[f'rama_pred_{cat}_fav'] = rama_pr_stats.get(cat,{}).get('favored',0)
                            result[f'rama_pred_{cat}_tot'] = rama_pr_stats.get(cat,{}).get('total',0)
                        result['rama_gt_pooled_fav'] = sum(c['favored'] for c in rama_gt_stats.values())
                        result['rama_gt_pooled_tot'] = sum(c['total'] for c in rama_gt_stats.values())
                        result['rama_pred_pooled_fav'] = sum(c['favored'] for c in rama_pr_stats.values())
                        result['rama_pred_pooled_tot'] = sum(c['total'] for c in rama_pr_stats.values())
                        result['pro_phi_mean_gt'] = float(np.mean(pro_gt)) if pro_gt else float('nan')
                        result['pro_phi_mean_pred'] = float(np.mean(pro_pr)) if pro_pr else float('nan')
                        result['pro_phi_n_gt'] = len(pro_gt)
                        result['pro_phi_n_pred'] = len(pro_pr)

                        bgt = _bond_stats(bonds_gt); bpr = _bond_stats(bonds_pr)
                        for bt in ['CA_C','N_CA','C_N']:
                            # Fix: wrap numpy values with float() for JSON serialization
                            result[f'bond_gt_mean_{bt}'] = float(bgt[bt][0]); result[f'bond_gt_std_{bt}'] = float(bgt[bt][1])
                            result[f'bond_pred_mean_{bt}'] = float(bpr[bt][0]); result[f'bond_pred_std_{bt}'] = float(bpr[bt][1])

                        # Per-residue-type SC RMSD
                        for res3 in sorted(sc_rmsd_by_type.keys()):
                            result[f'sc_rmsd_{res3}'] = sc_rmsd_by_type[res3]

                        results.append(result)

                        # Save cache
                        np.savez_compressed(
                            os.path.join(CACHE, f'{cache_key}.npz'),
                            xyz_pred=pred_np, xyz_gt=gt_np, rn=np.array(interior_names))
                        import pickle as _pk
                        with open(os.path.join(CACHE, f'{cache_key}_top.pkl'), 'wb') as _tf:
                            _pk.dump(top, _tf)

                except Exception as e:
                    print(f"{log_prefix} {ped_id} conf {mi}: error: {type(e).__name__}: {str(e)[:200]}")

            if os.path.exists(all_pdb):
                os.remove(all_pdb)

        # Per-system summary
        n_ok = sum(1 for r in results if r['ped_id'] == ped_id)
        if n_ok > 0:
            print(f"{log_prefix} {ped_id}: {n_ok}/{len(valid_indices)} conformers OK", flush=True)

        # Free GPU memory
        torch.cuda.empty_cache()
        gc.collect()

    return results

# -- Main ------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--systems', type=str, default=None, help='Comma-separated PED IDs')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--seed-offset', type=int, default=0, help='Seed offset for parallel workers')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    parser.add_argument('--max-conf', type=int, default=None, help='Max conformers per system (default: 500)')
    parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE, help='Conformers per load_dataset call')
    args = parser.parse_args()

    CHUNK_SIZE = args.chunk_size
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Load system list
    if args.systems:
        system_list = args.systems.split(',')
    else:
        with open(os.path.join(PROJECT, 'dataset-26-5-10/clean_v5.csv')) as f:
            system_list = sorted({r['ped_id'] for r in csv.DictReader(f)})

    ALL_PEDS = discover_peds()
    print(f"GPU{args.gpu}: {len(system_list)} systems: {system_list}", flush=True)

    results = process_systems(system_list, args.gpu, args.seed_offset, max_conf=args.max_conf)

    # Fix: convert numpy types to native Python types for JSON serialization
    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    out_path = args.output or os.path.join(OUT, f'results_gpu{args.gpu}.json')
    with open(out_path, 'w') as f:
        json.dump(sanitize(results), f)
    print(f"GPU{args.gpu}: {len(results)} conformers saved to {out_path}", flush=True)
