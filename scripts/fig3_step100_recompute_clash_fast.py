"""Recompute clash counts for all cached frames (24-core parallel).

Updates per_frame_metrics.csv clash column.
"""
import os, sys, pickle, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import mdtraj as md
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT = '/MDdata/data04/jxhuang/cg_cascade'
CACHE = os.path.join(PROJECT, 'logs/figure3_step100/cache')
OUT = os.path.join(PROJECT, 'logs/figure3_step100')
N_WORKERS = 24

VDW_RADII = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80}


def compute_clash(xyz, top):
    traj = md.Trajectory(xyz[np.newaxis, ...] / 10.0, topology=top)

    bonded = set()
    for bond in top.bonds:
        a1, a2 = bond[0].index, bond[1].index
        bonded.add((min(a1, a2), max(a1, a2)))
    atom_neighbors = defaultdict(set)
    for bond in top.bonds:
        a1, a2 = bond[0].index, bond[1].index
        atom_neighbors[a1].add(a2); atom_neighbors[a2].add(a1)
    for a in range(top.n_atoms):
        for n1 in atom_neighbors[a]:
            for n2 in atom_neighbors[a]:
                if n1 < n2:
                    bonded.add((n1, n2))

    heavy_idx = list(range(top.n_atoms))
    heavy_pairs = top.select_pairs(heavy_idx, heavy_idx)
    n_clash = 0
    if len(heavy_pairs) > 0:
        dists = md.compute_distances(traj, heavy_pairs)[0] * 10.0
        for pi in range(len(heavy_pairs)):
            a1, a2 = int(heavy_pairs[pi, 0]), int(heavy_pairs[pi, 1])
            if (min(a1, a2), max(a1, a2)) in bonded:
                continue
            e1 = top.atom(a1).element.symbol
            e2 = top.atom(a2).element.symbol
            r_vdw = VDW_RADII.get(e1, 1.70) + VDW_RADII.get(e2, 1.70)
            if r_vdw - dists[pi] > 0.4:
                n_clash += 1
    return n_clash


def clash_worker(task):
    row_idx, ped_id, frame_idx = task
    key = f'{ped_id}_{frame_idx}'
    cache_npz = os.path.join(CACHE, f'{key}.npz')
    cache_top = os.path.join(CACHE, f'{key}_top.pkl')

    if not os.path.exists(cache_npz) or not os.path.exists(cache_top):
        return row_idx, 0

    data = np.load(cache_npz, allow_pickle=True)
    xyz = data['xyz_recon']
    with open(cache_top, 'rb') as f:
        top = pickle.load(f)

    return row_idx, compute_clash(xyz, top)


# ── Main ──────────────────────────────────────────────────────────────────────
metrics_path = os.path.join(OUT, 'per_frame_metrics.csv')
df = pd.read_csv(metrics_path)
print(f'Loaded {len(df)} rows', flush=True)
print(f'Current clash: mean={df["clash"].mean():.2f}, max={df["clash"].max()}', flush=True)

# Build task list
tasks = []
for i, row in df.iterrows():
    ped_id = row['ped_id']
    frame_idx = int(row['frame_idx'])
    tasks.append((i, ped_id, frame_idx))

print(f'Tasks: {len(tasks)}, Workers: {N_WORKERS}', flush=True)

# Run parallel
t0 = time.time()
clash_values = np.zeros(len(df), dtype=int)
done = 0
total = len(tasks)

with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
    futures = {executor.submit(clash_worker, t): t for t in tasks}
    for future in as_completed(futures):
        row_idx, n_clash = future.result()
        clash_values[row_idx] = n_clash
        done += 1
        if done % 500 == 0:
            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (total - done) / rate
            mean_so_far = clash_values[:done].mean()
            print(f'  [{done}/{total}] {elapsed:.0f}s, {rate:.1f}/s, ETA {eta:.0f}s, mean_clash={mean_so_far:.1f}', flush=True)

elapsed = time.time() - t0
df['clash'] = clash_values
df.to_csv(metrics_path, index=False)

print(f'\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min)', flush=True)
print(f'New clash: mean={df["clash"].mean():.2f}, std={df["clash"].std():.2f}, max={df["clash"].max()}, median={df["clash"].median():.0f}', flush=True)

# Per-system summary
for ped_id in sorted(df['ped_id'].unique()):
    sub = df[df['ped_id'] == ped_id]
    print(f'  {ped_id}: clash mean={sub["clash"].mean():.1f}, std={sub["clash"].std():.1f}, max={sub["clash"].max()}')
