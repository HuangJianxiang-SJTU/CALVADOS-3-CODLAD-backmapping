#!/usr/bin/env python3
"""Folded-domain pilot benchmark for reviewer response."""
import os, sys, pickle, gc, argparse
from pathlib import Path

PROJECT = Path('/MDdata/data04/jxhuang/cg_cascade')
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT / 'scripts'))
sys.path.insert(0, str(PROJECT / 'src/CODLAD'))
sys.path.insert(0, str(PROJECT / 'src'))
sys.path.insert(0, str(PROJECT / 'src/cascade_codlad/eval_ensemble'))

import numpy as np
import pandas as pd
import mdtraj as md
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

import fig3_step100_worker as w

ZEN = PROJECT / 'data/raw_benchmark/_2024_Cao_CALVADOSCOM_Zenodo'
CODLAD_CKPT = PROJECT / 'legacy/misc/results-codlad'
OUT = PROJECT / 'logs/folded_pilot_rebuttal'
CACHE = OUT / 'cache'
TMP = PROJECT / 'data/folded_pilot_tmp'
FIGDIR = PROJECT / 'manuscript/figures'
TABDIR = PROJECT / 'manuscript/tables'
for p in [OUT, CACHE, TMP, FIGDIR, TABDIR]:
    p.mkdir(parents=True, exist_ok=True)

w.TMP = str(TMP)
w.CACHE = str(CACHE)
w.OUT = str(OUT)

SYSTEMS = {
    'Ubq2': {
        'ref': ZEN / 'src/extract_relax/Ubq2_rank0_relax.pdb',
        'cg_dcd': ZEN / 'data/IDPs_MDPsCOM_2.2_0.08_2_validate/Ubq2/3/Ubq2.dcd',
        'cg_top': ZEN / 'data/IDPs_MDPsCOM_2.2_0.08_2_validate/Ubq2/3/Ubq2_first.pdb',
    },
    'Gal3': {
        'ref': ZEN / 'src/extract_relax/Gal3_rank0_relax.pdb',
        'cg_dcd': ZEN / 'data/IDPs_MDPsCOM_2.2_0.08_2_validate/Gal3/3/Gal3.dcd',
        'cg_top': ZEN / 'data/IDPs_MDPsCOM_2.2_0.08_2_validate/Gal3/3/Gal3_first.pdb',
    },
}


def load_codlad_models(device='cuda:0', steps=100):
    """Load the same PED-trained CODLAD models as the IDP benchmark.

    The active CODLAD result symlinks in this checkout point to a missing
    top-level results-codlad directory, while the checkpoints are preserved
    under legacy/misc/results-codlad.
    """
    cwd = os.getcwd()
    os.chdir(w.CODLAD_DIR)
    try:
        from utils.model_module import get_vae_model
        vae_model, _ = get_vae_model(
            "N6",
            modelpath=str(CODLAD_CKPT / 'Vae_vqvae_ns36_vq3_vq4096'),
            device=device,
            modelnum=-1,
        )
        cvae_model, _ = get_vae_model("C2", device=device, modelnum=-1)
        vae_model.eval()
        cvae_model.eval()

        from models.latent_model import MPNN_models
        diff_model = MPNN_models['mpnn_diffusion'](
            input_size=3,
            unconditional=False,
            diffusion='diffusion',
            self_condition=False,
        ).to(device)
        ckpt = torch.load(
            str(CODLAD_CKPT / 'Diff_PED_mpnnnew/protein_weights_best.pt'),
            map_location='cpu',
        )
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
        diffusion = create_diffusion(
            str(steps),
            noise_schedule='linear',
            predict_xstart=False,
            rescale_learned_sigmas=False,
            self_condition=False,
        )
    finally:
        os.chdir(cwd)
    return vae_model, cvae_model, diff_model, diffusion


def heavy_trajectory(path):
    traj = md.load(str(path))
    heavy = traj.topology.select('element != H')
    return traj.atom_slice(heavy)


def ca_xyz_and_resnames(traj):
    ca = traj.topology.select('name CA')
    atoms = [traj.topology.atom(i) for i in ca]
    return traj.xyz[0, ca, :] * 10.0, [a.residue.name for a in atoms]


def ca_aligned_rmsd_A(xyz, top, ref_ca_A):
    ca = top.select('name CA')
    if len(ca) == 0:
        return np.nan
    mob = xyz[ca].astype(float)
    ref = ref_ca_A.astype(float)
    n = min(len(mob), len(ref))
    best = np.nan
    for mo in range(max(1, len(mob) - n + 1)):
        for ro in range(max(1, len(ref) - n + 1)):
            mob0 = mob[mo:mo + n] - mob[mo:mo + n].mean(axis=0)
            ref0 = ref[ro:ro + n] - ref[ro:ro + n].mean(axis=0)
            C = mob0.T @ ref0
            V, _, Wt = np.linalg.svd(C)
            d = np.sign(np.linalg.det(V @ Wt))
            U = V @ np.diag([1, 1, d]) @ Wt
            rmsd = float(np.sqrt(np.mean(np.sum((mob0 @ U - ref0) ** 2, axis=1))))
            best = rmsd if np.isnan(best) else min(best, rmsd)
    return best


def metric_row(system, condition, frame_idx, xyz_A, top, ref_ca_A=None):
    rg, dmax = w.measure_rg_dmax(xyz_A, top)
    n_fav, _, _, n_rot = w.measure_rotamer(xyz_A, top)
    cb = w.measure_cbdev(xyz_A, top)
    bonds = w.measure_bond_geometry(xyz_A, top)
    rama_rows, pro_phis = w.measure_ramachandran(xyz_A, top, f'{system}_{condition}_{frame_idx}')
    clash = w.measure_clash(xyz_A, top, f'{system}_{condition}_{frame_idx}')
    h, e, _, ss_tot = w.measure_dssp(xyz_A, top, f'{system}_{condition}_{frame_idx}')
    rama_fav = sum(1 for r in rama_rows if r.get('classification') == 'favored')
    rama_tot = sum(1 for r in rama_rows if r.get('classification') in ('favored', 'allowed', 'outlier'))
    return {
        'system': system,
        'condition': condition,
        'frame_idx': frame_idx,
        'n_residues': top.n_residues,
        'rg': rg,
        'dmax': dmax,
        'ca_rmsd_to_ref': ca_aligned_rmsd_A(xyz_A, top, ref_ca_A) if ref_ca_A is not None else 0.0,
        'rama_favored_pct': 100.0 * rama_fav / rama_tot if rama_tot else np.nan,
        'rotamer_favored_pct': 100.0 * n_fav / n_rot if n_rot else np.nan,
        'cbdev_mean': float(np.mean(cb)) if cb else np.nan,
        'clash_count': clash,
        'helix_frac': h / ss_tot if ss_tot else np.nan,
        'sheet_frac': e / ss_tot if ss_tot else np.nan,
        'pro_phi_mean': float(np.mean(pro_phis)) if pro_phis else np.nan,
        'bond_CA_C': float(bonds.get('CA_C', np.nan)),
        'bond_N_CA': float(bonds.get('N_CA', np.nan)),
        'bond_C_N': float(bonds.get('C_N', np.nan)),
    }


def run_reconstruct(cache_key, ca_A, resnames, models, device):
    npz = CACHE / f'{cache_key}.npz'
    top_pkl = CACHE / f'{cache_key}_top.pkl'
    if npz.exists() and top_pkl.exists():
        data = np.load(npz, allow_pickle=True)
        with open(top_pkl, 'rb') as f:
            top = pickle.load(f)
        return data['xyz_recon'], top
    tmp = TMP / f'{cache_key}_{os.getpid()}.pdb'
    w.build_pdb_from_ca(ca_A, resnames, str(tmp))
    cwd = os.getcwd()
    os.chdir(w.CODLAD_DIR)
    try:
        xyz, _, top = w.run_codlad_pipeline(str(tmp)[:-4], *models, device)
    finally:
        os.chdir(cwd)
        if tmp.exists():
            tmp.unlink()
    if xyz is None:
        raise RuntimeError(f'CODLAD failed for {cache_key}')
    np.savez(npz, xyz_recon=xyz, ca_input=ca_A)
    with open(top_pkl, 'wb') as f:
        pickle.dump(top, f)
    return xyz, top


def summarize(df):
    rows = []
    metrics = ['rg', 'dmax', 'ca_rmsd_to_ref', 'rama_favored_pct', 'rotamer_favored_pct',
               'cbdev_mean', 'clash_count', 'helix_frac', 'sheet_frac', 'pro_phi_mean',
               'bond_CA_C', 'bond_N_CA', 'bond_C_N']
    for (system, condition), sub in df.groupby(['system', 'condition']):
        row = {'system': system, 'condition': condition, 'n_frames': len(sub),
               'n_residues': int(sub.n_residues.iloc[0])}
        for m in metrics:
            row[f'{m}_mean'] = float(sub[m].mean())
            row[f'{m}_std'] = float(sub[m].std(ddof=1)) if len(sub) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def decomposition(summary):
    metric_map = {
        'CA RMSD to ref (A)': 'ca_rmsd_to_ref_mean',
        'Helix fraction': 'helix_frac_mean',
        'Sheet fraction': 'sheet_frac_mean',
        'Cbeta deviation (A)': 'cbdev_mean_mean',
        'Rama favored (%)': 'rama_favored_pct_mean',
        'Rotamer favored (%)': 'rotamer_favored_pct_mean',
        'Clash count': 'clash_count_mean',
        'Rg (A)': 'rg_mean',
        'Trans-Pro phi (deg)': 'pro_phi_mean_mean',
    }
    rows = []
    for system, sub in summary.groupby('system'):
        vals = {r.condition: r for _, r in sub.iterrows()}
        for label, col in metric_map.items():
            if not all(k in vals for k in ['Folded reference', 'Reference+CODLAD', 'Restrained CG+CODLAD']):
                continue
            ref = vals['Folded reference'][col]
            ped = vals['Reference+CODLAD'][col]
            cg = vals['Restrained CG+CODLAD'][col]
            rows.append({'system': system, 'metric': label, 'folded_reference': ref,
                         'reference_codlad': ped, 'restrained_cg_codlad': cg,
                         'pipeline_delta': ped - ref, 'cg_associated_delta': cg - ped,
                         'total_delta': cg - ref})
    return pd.DataFrame(rows)


def plot(decomp):
    summary = pd.read_csv(OUT / 'folded_pilot_per_condition_summary.csv')
    colors = {'Folded reference': '#6B7280', 'Reference+CODLAD': '#2F6B9A',
              'Restrained CG+CODLAD': '#C65D1E'}
    systems = [s for s in ['Ubq2', 'Gal3'] if s in set(summary.system)]
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.9), constrained_layout=True)

    ax = axes[0]
    pos, vals, cols, centers, labels, k = [], [], [], [], [], 0
    for system in systems:
        for metric, label in [('helix_frac_mean', 'helix'), ('sheet_frac_mean', 'sheet')]:
            for condition in ['Folded reference', 'Reference+CODLAD']:
                row = summary[(summary.system == system) & (summary.condition == condition)].iloc[0]
                pos.append(k)
                vals.append(row[metric])
                cols.append(colors[condition])
                k += 1
            centers.append(k - 1.5)
            labels.append(f'{system}\n{label}')
            k += 0.7
    ax.bar(pos, vals, color=cols, width=0.72)
    ax.set_xticks(centers)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 0.38)
    ax.set_ylabel('DSSP fraction')
    ax.set_title('A. Folded-reference reconstruction', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)

    ax = axes[1]
    cg = summary[summary.condition == 'Restrained CG+CODLAD'].set_index('system')
    ca = [cg.loc[s, 'ca_rmsd_to_ref_mean'] for s in systems]
    rg_delta = []
    for system in systems:
        ref = summary[(summary.system == system) &
                      (summary.condition == 'Folded reference')].iloc[0]['rg_mean']
        cgr = summary[(summary.system == system) &
                      (summary.condition == 'Restrained CG+CODLAD')].iloc[0]['rg_mean']
        rg_delta.append(cgr - ref)
    x = np.arange(len(systems))
    ax.bar(x - 0.18, ca, width=0.35, color='#C65D1E', label='CA/bead RMSD')
    ax.bar(x + 0.18, rg_delta, width=0.35, color='#D9A441', label='Rg shift')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.set_ylabel('Angstrom')
    ax.set_title('B. Restrained MDP input differs', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=7, loc='upper left')

    ax = axes[2]
    pos, vals, cols, centers, labels, k = [], [], [], [], [], 0
    for system in systems:
        for metric, label in [('rama_favored_pct_mean', 'Rama'),
                              ('rotamer_favored_pct_mean', 'Rotamer')]:
            for condition in ['Folded reference', 'Reference+CODLAD', 'Restrained CG+CODLAD']:
                row = summary[(summary.system == system) & (summary.condition == condition)].iloc[0]
                pos.append(k)
                vals.append(row[metric])
                cols.append(colors[condition])
                k += 1
            centers.append(k - 2)
            labels.append(f'{system}\n{label}')
            k += 0.8
    ax.bar(pos, vals, color=cols, width=0.72)
    ax.set_xticks(centers)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(65, 102)
    ax.set_ylabel('Favored residues (%)')
    ax.set_title('C. Local quality metrics', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=colors[k], label=k)
               for k in ['Folded reference', 'Reference+CODLAD', 'Restrained CG+CODLAD']]
    fig.legend(handles=handles, frameon=False, fontsize=8, loc='lower center',
               ncol=3, bbox_to_anchor=(0.5, -0.03))
    for ext in ['png', 'svg']:
        fig.savefig(FIGDIR / f'folded_pilot_rebuttal.{ext}', dpi=300, bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--systems', default='Ubq2,Gal3')
    parser.add_argument('--n-frames', type=int, default=20)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('device', device, 'systems', args.systems, 'n_frames', args.n_frames, flush=True)
    models = load_codlad_models(device=device, steps=100)
    rows = []
    for system in args.systems.split(','):
        cfg = SYSTEMS[system]
        print('\\n==', system, flush=True)
        ref = heavy_trajectory(cfg['ref'])
        ref_ca_A, ref_resnames = ca_xyz_and_resnames(ref)
        rows.append(metric_row(system, 'Folded reference', 0, ref.xyz[0] * 10.0, ref.topology, ref_ca_A))
        xyz_ref, top_ref = run_reconstruct(f'{system}_refcodlad', ref_ca_A, ref_resnames, models, device)
        rows.append(metric_row(system, 'Reference+CODLAD', 0, xyz_ref, top_ref, ref_ca_A))
        traj = md.load(str(cfg['cg_dcd']), top=str(cfg['cg_top']))
        rnames = [r.name for r in traj.topology.residues]
        indices = np.linspace(0, traj.n_frames - 1, min(args.n_frames, traj.n_frames), dtype=int)
        for j, frame_idx in enumerate(indices):
            ca_A = traj.xyz[frame_idx] * 10.0
            xyz, top = run_reconstruct(f'{system}_cg_{int(frame_idx)}', ca_A, rnames, models, device)
            rows.append(metric_row(system, 'Restrained CG+CODLAD', int(frame_idx), xyz, top, ref_ca_A))
            print(system, j + 1, '/', len(indices), 'frame', int(frame_idx), flush=True)
            torch.cuda.empty_cache()
            gc.collect()
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'folded_pilot_per_frame_metrics.csv', index=False)
    summary = summarize(df)
    summary.to_csv(OUT / 'folded_pilot_per_condition_summary.csv', index=False)
    decomp = decomposition(summary)
    decomp.to_csv(TABDIR / 'folded_pilot_decomposition_rebuttal.csv', index=False)
    plot(decomp)
    print('\\nSUMMARY')
    print(summary.round(3).to_string(index=False))
    print('\\nDECOMP')
    print(decomp.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
