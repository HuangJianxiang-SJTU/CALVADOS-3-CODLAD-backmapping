"""Recompute clash counts for all cached frames using PDBFixer + MolProbity-style detection.
Runs in parallel with 24 CPU workers. Updates per_frame_metrics.csv clash column."""
import os, sys, pickle, time, warnings, tempfile
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import mdtraj as md
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT = '/MDdata/data04/jxhuang/cg_cascade'
CACHE = os.path.join(PROJECT, 'logs/figure3_step100/cache')
OUT = os.path.join(PROJECT, 'logs/figure3_step100')
TMP_BASE = os.path.join(PROJECT, 'data/fig3_clash_tmp')
N_WORKERS = 24

os.makedirs(TMP_BASE, exist_ok=True)

VDW_RADII = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80}

def clash_worker(task):
    """Compute clash for one frame. task = (ped_id, frame_idx)."""
    ped_id, frame_idx = task
    key = f'{ped_id}_{frame_idx}'
    cache_npz = os.path.join(CACHE, f'{key}.npz')
    cache_top = os.path.join(CACHE, f'{key}_top.pkl')

    try:
        data = np.load(cache_npz, allow_pickle=True)
        xyz = data['xyz_recon']
        with open(cache_top, 'rb') as f:
            top = pickle.load(f)
    except Exception:
        return (ped_id, frame_idx, -1)

    # Save heavy-atom PDB and add H
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile

    heavy_pdb = os.path.join(TMP_BASE, f'{key}_heavy.pdb')
    h_pdb = os.path.join(TMP_BASE, f'{key}_h.pdb')

    try:
        traj_heavy = md.Trajectory(xyz[np.newaxis, ...] / 10.0, topology=top)
        traj_heavy.save(heavy_pdb)
        fixer = PDBFixer(filename=heavy_pdb)
        fixer.addMissingHydrogens(pH=7.0)
        with open(h_pdb, 'w') as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)

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
    except Exception:
        n_clash = -1  # flag failure

    for p in [heavy_pdb, h_pdb]:
        if os.path.exists(p): os.remove(p)
    return (ped_id, frame_idx, n_clash)

# Build task list
metrics_path = os.path.join(OUT, 'per_frame_metrics.csv')
df = pd.read_csv(metrics_path)
print(f'Loaded {len(df)} rows from {metrics_path}')
print(f'Current clash: mean={df["clash"].mean():.2f}, max={df["clash"].max()}')

# Only recompute frames where clash == 0 (the stubbed values)
tasks = []
for _, row in df[df['clash'] == 0].iterrows():
    tasks.append((row['ped_id'], int(row['frame_idx'])))

print(f'Frames to recompute: {len(tasks)}')
print(f'Workers: {N_WORKERS}')

if tasks:
    t0 = time.time()
    done = 0
    results = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(clash_worker, t): t for t in tasks}
        for future in as_completed(futures):
            ped_id, frame_idx, n_clash = future.result()
            results[(ped_id, frame_idx)] = n_clash
            done += 1
            if done % 500 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                rate = done / elapsed * 60
                print(f'  {done}/{len(tasks)} ({done/len(tasks)*100:.0f}%), '
                      f'{rate:.0f} frames/min, ETA {((len(tasks)-done)/rate):.0f} min', flush=True)
    print(f'  Clash recomputation done in {(time.time()-t0)/60:.1f} min')

    # Update DataFrame
    n_updated = 0
    n_failed = 0
    for (ped_id, frame_idx), n_clash in results.items():
        mask = (df['ped_id'] == ped_id) & (df['frame_idx'] == frame_idx)
        if mask.any():
            if n_clash >= 0:
                df.loc[mask, 'clash'] = n_clash
                n_updated += 1
            else:
                n_failed += 1

    print(f'Updated {n_updated} rows, {n_failed} failures')

    # Save
    df.to_csv(metrics_path, index=False)
    print(f'Saved updated {metrics_path}')
    print(f'New clash: mean={df["clash"].mean():.2f}, max={df["clash"].max()}, median={df["clash"].median():.0f}')

    # Per-system summary
    for ped_id in sorted(df['ped_id'].unique()):
        sub = df[df['ped_id'] == ped_id]
        print(f'  {ped_id}: clash mean={sub["clash"].mean():.1f}, std={sub["clash"].std():.1f}, max={sub["clash"].max()}')
else:
    print('No frames need recomputation')
