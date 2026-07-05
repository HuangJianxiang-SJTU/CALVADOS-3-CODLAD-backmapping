"""Figure 3 Step=100: CODLAD pipeline on CALVADOS 3 CG trajectories.

Pipeline: CALVADOS CA -> build_pdb_from_ca -> CVAE(C2) -> MPNN diffusion(100 steps) -> VAE(N6) decode.
Computes structural metrics and CA fidelity for each frame.
Measures degradation attributable to CG input (vs Figure 2 PED+CODLAD).

Usage:
  python scripts/fig3_step100_worker.py --gpu 0 --systems PED00074
  python scripts/fig3_step100_worker.py --gpu 0  # all 23 clean_v5 systems
"""
import os, sys, csv, json, random, warnings, time, io, pickle, gc
warnings.filterwarnings("ignore")

os.chdir('/MDdata/data04/jxhuang/cg_cascade')
sys.path.insert(0, 'src/CODLAD'); sys.path.insert(0, 'src')
sys.path.insert(0, 'src/cascade_codlad/eval_ensemble')

import numpy as np
import torch
import mdtraj as md
import pandas as pd
from collections import defaultdict

from utils.utils_ic import ic_to_xyz, core_atoms as CODLAD_CORE_ATOMS
from metrics_ramachandran import _in_boxes, _get_region_boxes

# -- Configuration ---------------------------------------------------------------
PROJECT = '/MDdata/data04/jxhuang/cg_cascade'
CODLAD_DIR = os.path.join(PROJECT, 'src/CODLAD')
CG_DIR = os.path.join(PROJECT, 'dataset-26-5-10/cg_simulations')
OUT = os.path.join(PROJECT, 'logs/figure3_step100')
CACHE = os.path.join(OUT, 'cache')
TMP = os.path.join(PROJECT, 'data/fig3_step100_tmp')
SEED = 42
STEPS = 100      # diffusion steps (author default)
N_FRAMES = 500   # every 8th from 4000-frame trajectory

os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)
os.makedirs(TMP, exist_ok=True)

COD_IDX2RES = {
    0:'ASN',1:'HIS',2:'ALA',3:'GLY',4:'ARG',5:'MET',6:'SER',7:'ILE',
    8:'GLU',9:'LEU',10:'TYR',11:'ASP',12:'VAL',13:'TRP',14:'GLN',
    15:'LYS',16:'PRO',17:'PHE',18:'CYS',19:'THR',
}

CANONICAL_BONDS = {'CA_C': 1.53, 'N_CA': 1.46, 'C_N': 1.33}

# -- Model loading ---------------------------------------------------------------

def load_models(device='cuda:0', steps=100):
    _cwd = os.getcwd()
    os.chdir(CODLAD_DIR)
    try:
        from utils.model_module import get_vae_model
        vae_model, _ = get_vae_model("N6", device=device, modelnum=-1)
        cvae_model, _ = get_vae_model("C2", device=device, modelnum=-1)
        vae_model.eval(); cvae_model.eval()

        from models.latent_model import MPNN_models
        diff_model = MPNN_models['mpnn_diffusion'](
            input_size=3, unconditional=False, diffusion='diffusion',
            self_condition=False).to(device)
        ckpt = torch.load('results/Diff_PED_mpnnnew/protein_weights_best.pt',
                          map_location='cpu')
        try:
            diff_model.load_state_dict(ckpt['net_model'])
        except RuntimeError:
            from collections import OrderedDict
            ns = OrderedDict()
            for k, v in ckpt['net_model'].items():
                ns[k[7:]] = v
            diff_model.load_state_dict(ns)
        diff_model.eval()

        from diffusion_and_flow import create_diffusion
        diffusion = create_diffusion(str(steps), noise_schedule='linear',
                                      predict_xstart=False,
                                      rescale_learned_sigmas=False,
                                      self_condition=False)
    finally:
        os.chdir(_cwd)
    return vae_model, cvae_model, diff_model, diffusion

# -- PDB building from CA --------------------------------------------------------

def build_pdb_from_ca(ca_xyz_A, resnames, path):
    n = len(ca_xyz_A)
    coords = {}
    for ri in range(n):
        for aname in CODLAD_CORE_ATOMS.get(resnames[ri], CODLAD_CORE_ATOMS['ALA']):
            coords[(ri, aname)] = ca_xyz_A[ri].copy()
    for ri in range(1, n - 1):
        ca = ca_xyz_A[ri]
        vp = ca_xyz_A[ri - 1] - ca
        vn = ca_xyz_A[ri + 1] - ca
        dp = np.linalg.norm(vp)
        dn = np.linalg.norm(vn)
        if dp > 1e-6:
            coords[(ri, 'N')] = ca + vp / dp * 1.46
        if dn > 1e-6:
            coords[(ri, 'C')] = ca + vn / dn * 1.53
        npos = coords.get((ri, 'N'), ca)
        cpos = coords.get((ri, 'C'), ca)
        normal = np.cross(npos - ca, cpos - ca)
        nn = np.linalg.norm(normal)
        if nn > 1e-6:
            coords[(ri, 'O')] = ca + normal / nn * 1.23

    with open(path, 'w') as f:
        f.write("MODEL        1\n")
        ai = 0
        for ri in range(n):
            res3 = resnames[ri]
            for aname in CODLAD_CORE_ATOMS.get(res3, CODLAD_CORE_ATOMS['ALA']):
                xyz = coords.get((ri, aname), ca_xyz_A[ri])
                nm = f' {aname:<3s}' if len(aname) < 4 else f'{aname:<4s}'
                f.write(f'ATOM  {ai+1:5d} {nm} {res3} A{ri+1:4d}    '
                        f'{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {aname[0]:>2s}  \n')
                ai += 1
        f.write("END\n")

# -- CODLAD pipeline -------------------------------------------------------------

def run_codlad_pipeline(pdb_stem_abs, vae_model, cvae_model, diff_model, diffusion, device):
    from utils.dataset_module import load_dataset, get_norm_feature
    from utils.train_module import batch_to as _batch_to
    DATA_PARAMS = {"atom_cutoff": 9.0, "cg_cutoff": 21.0, "edgeorder": 2}

    _stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        loader, info, n_atoms, n_cgs, _, _top = load_dataset(pdb_stem_abs, DATA_PARAMS, single=True)
    finally:
        sys.stdout.close()
        sys.stdout = _stdout

    with torch.no_grad():
        for batch in loader:
            batch = _batch_to(batch, device)
            nres = int(batch["num_CGs"][0]) + 2

            y, _, _, mask, num_CGs, mu, sigma = cvae_model.get_latent_cg(batch)
            seq_len = batch['num_CGs'][0].item()
            z = torch.randn((1, seq_len, 3), device=device)
            cat_z = torch.cat([z, z], 0)
            y_null = torch.zeros_like(y)
            cat_y = torch.cat([y, y_null], 0)
            cat_mask = torch.cat([mask, mask], 0)
            batch["randn"] = torch.randn([cat_z.shape[0], cat_z.shape[1]], device=device)
            model_kwargs = dict(y=cat_y, mask=cat_mask, batch=batch)

            samples = diffusion.p_sample_loop(
                diff_model.forward, cat_z.shape, cat_z, clip_denoised=False,
                model_kwargs=model_kwargs, progress=False, device=device)
            samples, _ = samples.chunk(2, dim=0)

            samples = get_norm_feature(samples, "N6", norm_channel=True, norm_single=False,
                                        norm_in=False, dataname="PED")

            ic, ic_recon = vae_model.latent_decode(samples, mask, batch)
            ic_recon = ic_recon.reshape(-1, nres - 2, 13, 3)
            OG = batch["OG_CG_nxyz"].reshape(-1, nres, 4)
            xyz_recon = ic_to_xyz(OG, ic_recon, info[0]).reshape(-1, 3)

            cg_types = batch["CG_nxyz"][:, 0].long().cpu().numpy()
            rn = [COD_IDX2RES.get(int(t), 'ALA') for t in cg_types]
            result = (xyz_recon.cpu().numpy(), rn, _top)
            del batch, y, mask, num_CGs, mu, sigma, z, cat_z, cat_y, cat_mask
            del samples, ic, ic_recon, OG, xyz_recon, cg_types
            del loader, info, n_atoms, n_cgs
            gc.collect()
            torch.cuda.empty_cache()
            return result
    return None, None, None

# -- Metric functions (with all bug fixes) ----------------------------------------

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
    except Exception:
        for p in [heavy_pdb, h_pdb]:
            if os.path.exists(p): os.remove(p)
        return None
    if os.path.exists(heavy_pdb): os.remove(heavy_pdb)
    return h_pdb

def measure_rg_dmax(xyz, top):
    traj = md.Trajectory(xyz[np.newaxis, ...] / 10.0, topology=top)
    rg = float(md.compute_rg(traj)[0]) * 10.0  # nm -> Angstroms
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

# -- Main worker -----------------------------------------------------------------

def process_systems(ped_ids, gpu_id, seed_offset=0):
    DEVICE = 'cuda:0'
    random.seed(SEED + gpu_id + seed_offset)
    np.random.seed(SEED + gpu_id + seed_offset)
    torch.manual_seed(SEED + gpu_id + seed_offset)

    vae_model, cvae_model, diff_model, diffusion = load_models(device=DEVICE, steps=STEPS)
    log_prefix = f"[GPU{gpu_id}]"
    results = []

    for pi, ped_id in enumerate(ped_ids):
        dcd = os.path.join(CG_DIR, ped_id, f'{ped_id}.dcd')
        top_pdb = dcd.replace('.dcd', '_first.pdb')
        if not os.path.exists(dcd):
            print(f"{log_prefix} {ped_id}: no CG trajectory, skip"); continue
        print(f"{log_prefix} [{pi+1}/{len(ped_ids)}] {ped_id}: starting", flush=True)

        traj = md.load(dcd, top=top_pdb)
        rnames = [r.name for r in traj.topology.residues]
        n_total = traj.n_frames

        # Sample 500 frames: every 8th from the 4000-frame trajectory
        frame_indices = list(range(0, min(n_total, 4000), 8))[:N_FRAMES]
        n_sample = len(frame_indices)
        ca_A_all = traj.xyz[frame_indices] * 10  # nm -> Angstroms
        print(f"{log_prefix} {ped_id}: {n_total} frames, sampling {n_sample}", flush=True)

        per_frame_rows = []
        n_success = 0; n_fail = 0; n_cached = 0
        t_start = time.time()
        oom_hit = False

        for fi, frame_idx in enumerate(frame_indices):
            cache_key = f'{ped_id}_{frame_idx}'
            cache_npz = os.path.join(CACHE, f'{cache_key}.npz')
            cache_top = os.path.join(CACHE, f'{cache_key}_top.pkl')

            if os.path.exists(cache_npz) and os.path.exists(cache_top):
                n_cached += 1
                continue  # skip cached frames; metrics already in per-frame CSV
            else:
                # Run CODLAD inference
                tmp_pdb = os.path.join(TMP, f'{ped_id}_{frame_idx}_{os.getpid()}.pdb')
                build_pdb_from_ca(ca_A_all[fi], rnames, tmp_pdb)
                pdb_stem_abs = tmp_pdb[:-4]

                torch.cuda.synchronize(DEVICE)
                torch.cuda.empty_cache()
                gc.collect()

                _cwd = os.getcwd()
                os.chdir(CODLAD_DIR)
                result = None
                try:
                    result = run_codlad_pipeline(pdb_stem_abs, vae_model, cvae_model,
                                                  diff_model, diffusion, DEVICE)
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or 'out of memory' in str(e).lower()
                    if is_oom:
                        print(f"{log_prefix} {ped_id} frame {frame_idx}: OOM - stopping this system")
                        torch.cuda.synchronize(DEVICE); torch.cuda.empty_cache()
                        gc.collect()
                        oom_hit = True
                        break
                    print(f"{log_prefix} {ped_id} frame {frame_idx}: {type(e).__name__}: {str(e)[:120]}")
                    n_fail += 1
                except Exception as e:
                    print(f"{log_prefix} {ped_id} frame {frame_idx}: {type(e).__name__}: {str(e)[:120]}")
                    torch.cuda.synchronize(DEVICE); torch.cuda.empty_cache()
                    n_fail += 1
                os.chdir(_cwd)

                if result is None:
                    if os.path.exists(tmp_pdb): os.remove(tmp_pdb)
                    continue

                if os.path.exists(tmp_pdb): os.remove(tmp_pdb)
                if result is None or result[0] is None:
                    n_fail += 1; continue

                xyz_r, rn_r, top = result
                ca_input = ca_A_all[fi]
                np.savez_compressed(cache_npz, xyz_recon=xyz_r, rn=np.array(rn_r),
                                    ca_input=ca_input)
                with open(cache_top, 'wb') as f:
                    pickle.dump(top, f)

                torch.cuda.synchronize(DEVICE)
                torch.cuda.empty_cache()
                gc.collect()

            # Compute metrics
            conf_key = f"{ped_id}_{frame_idx}"
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

            # Aggregate Ramachandran
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

            per_frame_rows.append(row)
            n_success += 1

            # Aggressive memory cleanup every 10 frames
            if (fi + 1) % 10 == 0:
                torch.cuda.synchronize(DEVICE)
                torch.cuda.empty_cache()
                gc.collect()

            if (fi + 1) % 50 == 0:
                elapsed = time.time() - t_start
                rate = (fi + 1) / elapsed
                print(f"{log_prefix} [{fi+1}/{n_sample}] {elapsed:.0f}s, cached={n_cached}, fail={n_fail}", flush=True)

        # Save per-frame metrics
        df = pd.DataFrame(per_frame_rows)
        per_frame_path = os.path.join(OUT, f'{ped_id}_per_frame.csv')
        df.to_csv(per_frame_path, index=False)
        oom_msg = " (OOM - will continue on next run)" if oom_hit else ""
        print(f"{log_prefix} {ped_id}: {n_success} success, {n_fail} fail, {n_cached} cached{oom_msg}", flush=True)

        results.extend(per_frame_rows)
        torch.cuda.empty_cache(); gc.collect()

        # If OOM hit, try to reset GPU state for next system
        if oom_hit:
            try:
                torch.cuda.synchronize(DEVICE)
            except Exception:
                pass  # sync may fail after OOM, that's ok
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(3)

    return results

# -- Main ------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--systems', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed-offset', type=int, default=0)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.systems:
        system_list = args.systems.split(',')
    else:
        with open(os.path.join(PROJECT, 'dataset-26-5-10/clean_v5.csv')) as f:
            system_list = sorted({r['ped_id'] for r in csv.DictReader(f)})

    print(f"GPU{args.gpu}: {len(system_list)} systems: {system_list}", flush=True)

    results = process_systems(system_list, args.gpu, args.seed_offset)

    def sanitize(obj):
        if isinstance(obj, dict): return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list): return [sanitize(v) for v in obj]
        elif isinstance(obj, (np.integer,)): return int(obj)
        elif isinstance(obj, (np.floating,)): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    out_path = args.output or os.path.join(OUT, f'results_gpu{args.gpu}.json')
    with open(out_path, 'w') as f:
        json.dump(sanitize(results), f)
    print(f"GPU{args.gpu}: {len(results)} frames saved to {out_path}", flush=True)
