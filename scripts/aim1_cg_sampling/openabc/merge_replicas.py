"""
Merge the 4 per-replica OpenABC DCDs for each system into a single 4000-frame
trajectory matching the CALVADOS layout: {PED}.dcd + {PED}_first.pdb (Cα, nm,
single chain). This is the exact contract scripts/fig3_step100_worker.py reads.

Also aggregates clock_time_rep{r}.json files into a per-system summary.

Usage (openabc env):
    python merge_replicas.py --model moff2
    python merge_replicas.py --model mpipi --systems PED00074,PED00003
"""
import argparse
import json
import os
from pathlib import Path

import mdtraj as md

PROJECT = '/MDdata/data04/jxhuang/cg_cascade'
N_REPLICAS = 4


def load_systems():
    import csv
    with open(os.path.join(PROJECT, 'dataset/clean_v5.csv')) as f:
        return [r['ped_id'] for r in csv.DictReader(f)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, choices=['mpipi', 'moff2'])
    p.add_argument('--n_replicas', type=int, default=N_REPLICAS)
    p.add_argument('--systems', default=None)
    args = p.parse_args()

    systems = args.systems.split(',') if args.systems else load_systems()
    base = Path(PROJECT) / f'dataset/cg_simulations_{args.model}'

    for ped_id in systems:
        sys_dir = base / ped_id
        out_dcd = sys_dir / f'{ped_id}.dcd'
        top_pdb = sys_dir / f'{ped_id}_first.pdb'

        # Topology: use rep0's first-frame PDB if present, else the input CA pdb
        if not top_pdb.exists():
            ca = Path(PROJECT) / f'dataset/cg_input_ca/{ped_id}/{ped_id}_ca.pdb'
            import shutil
            shutil.copyfile(ca, top_pdb)

        # Skip if merged DCD already exists with correct frame count
        rep_dcds = [sys_dir / f'{ped_id}_rep{r}.dcd' for r in range(args.n_replicas)]
        missing = [str(p) for p in rep_dcds if not p.exists()]
        if missing:
            print(f'[skip] {ped_id}: missing replicas {missing}')
            continue

        if out_dcd.exists():
            try:
                t = md.load(str(out_dcd), top=str(top_pdb))
                if t.n_frames == args.n_replicas * 1000:
                    print(f'[skip] {ped_id}: merged DCD already has {t.n_frames} frames')
                    continue
            except Exception:
                pass

        # Concatenate replicas (they share topology from the same CA input)
        trajs = [md.load(str(p), top=str(top_pdb)) for p in rep_dcds]
        merged = md.join(trajs)
        merged.save_dcd(str(out_dcd))

        # Aggregate clock times
        clocks = []
        for r in range(args.n_replicas):
            clk = sys_dir / f'clock_time_rep{r}.json'
            if clk.exists():
                clocks.append(json.load(open(clk)))
        agg = {
            'ped_id': ped_id, 'model': args.model,
            'n_replicas': len(clocks),
            'n_frames_total': int(sum(c['n_frames'] for c in clocks)),
            'wall_seconds_total': float(sum(c['wall_seconds'] for c in clocks)),
            'wall_seconds_mean_per_replica': float(sum(c['wall_seconds'] for c in clocks) / max(len(clocks), 1)),
            'n_residues': clocks[0]['n_residues'] if clocks else None,
            'box_nm': clocks[0]['box_nm'] if clocks else None,
            'per_replica': clocks,
        }
        json.dump(agg, open(sys_dir / 'clock_time.json', 'w'), indent=4)

        t = md.load(str(out_dcd), top=str(top_pdb))
        print(f'[ok] {ped_id}: merged {t.n_frames} frames, {t.n_atoms} atoms -> {out_dcd}')


if __name__ == '__main__':
    main()
