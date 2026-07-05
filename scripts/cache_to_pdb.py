"""Convert cached npz frames to PDB files (parallel).

Figure 2: xyz_pred → figure2_step100/pdbs/
Figure 3: xyz_recon → figure3_step100/pdbs/

Usage:
  python scripts/cache_to_pdb.py figure2   # convert figure2 cache
  python scripts/cache_to_pdb.py figure3   # convert figure3 cache
  python scripts/cache_to_pdb.py all       # convert both
"""
import os, sys, pickle, time
import numpy as np
import mdtraj as md
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT = '/MDdata/data04/jxhuang/cg_cascade'
N_WORKERS = 8  # moderate: I/O-bound, too many workers causes disk thrashing

CONFIGS = {
    'figure2': {
        'cache': os.path.join(PROJECT, 'logs/figure2_step100/cache'),
        'out_dir': os.path.join(PROJECT, 'logs/figure2_step100/pdbs'),
        'xyz_key': 'xyz_pred',
    },
    'figure3': {
        'cache': os.path.join(PROJECT, 'logs/figure3_step100/cache'),
        'out_dir': os.path.join(PROJECT, 'logs/figure3_step100/pdbs'),
        'xyz_key': 'xyz_recon',
    },
}


def convert_worker(task):
    cache_dir, out_dir, xyz_key, key = task
    npz_path = os.path.join(cache_dir, f'{key}.npz')
    top_path = os.path.join(cache_dir, f'{key}_top.pkl')
    pdb_path = os.path.join(out_dir, f'{key}.pdb')

    if os.path.exists(pdb_path):
        return key, True  # skip existing

    if not os.path.exists(npz_path) or not os.path.exists(top_path):
        return key, False

    try:
        data = np.load(npz_path, allow_pickle=True)
        xyz = data[xyz_key]
        with open(top_path, 'rb') as f:
            top = pickle.load(f)

        traj = md.Trajectory(xyz[np.newaxis, ...] / 10.0, topology=top)
        traj.save(pdb_path)
        return key, True
    except Exception as e:
        return key, False


def convert_figure(name, config):
    cache_dir = config['cache']
    out_dir = config['out_dir']
    xyz_key = config['xyz_key']

    os.makedirs(out_dir, exist_ok=True)

    # Collect keys from npz files
    keys = sorted([f[:-4] for f in os.listdir(cache_dir) if f.endswith('.npz')])
    print(f'Found {len(keys)} frames in {name} cache')

    # Check existing
    existing = set(f[:-4] for f in os.listdir(out_dir) if f.endswith('.pdb'))
    tasks = [(cache_dir, out_dir, xyz_key, k) for k in keys if k not in existing]
    skipped = len(keys) - len(tasks)
    if skipped:
        print(f'  {skipped} already exist, {len(tasks)} to convert')

    if not tasks:
        print(f'  All done.')
        return

    t0 = time.time()
    done = 0
    total = len(tasks)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(convert_worker, t): t for t in tasks}
        for future in as_completed(futures):
            key, ok = future.result()
            done += 1
            if done % 500 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (total - done) / rate
                print(f'  [{done}/{total}] {elapsed:.0f}s, {rate:.1f}/s, ETA {eta:.0f}s')

    elapsed = time.time() - t0
    print(f'  Done in {elapsed:.0f}s ({elapsed/60:.1f} min), {total} PDBs → {out_dir}')


if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else 'all'

    for name in ['figure2', 'figure3']:
        if target in (name, 'all'):
            print(f'\n{"="*60}')
            print(f'  Converting {name} cache → PDBs')
            print(f'{"="*60}')
            convert_figure(name, CONFIGS[name])
