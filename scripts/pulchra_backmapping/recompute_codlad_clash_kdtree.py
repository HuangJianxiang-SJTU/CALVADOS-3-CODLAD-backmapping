#!/usr/bin/env python3
"""Recompute CODLAD clash counts from cached coordinates using the 0.4 A vdW-overlap definition.

This is the same heavy-atom KDTree implementation used for the PULCHRA comparison.
It intentionally does not add hydrogens, matching the current batch-analysis path
and avoiding the fig3 bug where a stubbed hydrogen-addition function returned zero
clashes for every frame.
"""
import argparse
import os
import pickle
import shutil
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

PROJECT = Path('/MDdata/data04/jxhuang/cg_cascade')
VDW_RADII = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80}


def backup(path: Path):
    if not path.exists():
        return None
    bak = path.with_suffix(path.suffix + '.bak_pre_kdtree_clash')
    if not bak.exists():
        shutil.copy2(path, bak)
    return bak


def heavy_clash_count(xyz_A, top):
    heavy = [(a.index, a.element.symbol if a.element is not None else 'C')
             for a in top.atoms
             if a.element is None or a.element.symbol != 'H']
    if len(heavy) < 2:
        return 0

    heavy_idx = np.array([h[0] for h in heavy], dtype=int)
    elems = [h[1] for h in heavy]
    heavy_pos_nm = xyz_A[heavy_idx] / 10.0

    excluded = set()
    neighbors = defaultdict(set)
    for bond in top.bonds:
        a1, a2 = bond[0].index, bond[1].index
        excluded.add((min(a1, a2), max(a1, a2)))
        neighbors[a1].add(a2)
        neighbors[a2].add(a1)
    for a, ns in neighbors.items():
        ns = list(ns)
        for i in range(len(ns)):
            for j in range(i + 1, len(ns)):
                excluded.add((min(ns[i], ns[j]), max(ns[i], ns[j])))

    tree = cKDTree(heavy_pos_nm)
    # Max possible clash distance is S-S: 1.80 + 1.80 - 0.4 = 3.2 A = 0.32 nm.
    pairs = tree.query_pairs(r=0.32, output_type='ndarray')
    if len(pairs) == 0:
        return 0

    n_clash = 0
    for local_i, local_j in pairs:
        global_i = int(heavy_idx[local_i])
        global_j = int(heavy_idx[local_j])
        if (min(global_i, global_j), max(global_i, global_j)) in excluded:
            continue
        dist_A = float(np.linalg.norm(heavy_pos_nm[local_i] - heavy_pos_nm[local_j])) * 10.0
        r_vdw = VDW_RADII.get(elems[local_i], 1.70) + VDW_RADII.get(elems[local_j], 1.70)
        if r_vdw - dist_A > 0.4:
            n_clash += 1
    return int(n_clash)


def load_top(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def fig3_worker(task):
    row_idx, ped_id, frame_idx = task
    key = f'{ped_id}_{int(frame_idx)}'
    cache = PROJECT / 'logs/figure3_step100/cache'
    npz = cache / f'{key}.npz'
    top_p = cache / f'{key}_top.pkl'
    if not npz.exists() or not top_p.exists():
        return row_idx, np.nan
    data = np.load(npz, allow_pickle=True)
    top = load_top(top_p)
    return row_idx, heavy_clash_count(data['xyz_recon'], top)


def fig2_worker(task):
    row_idx, ped_id, conformer_idx = task
    key = f'{ped_id}_{int(conformer_idx)}'
    cache = PROJECT / 'logs/figure2_step100/cache'
    npz = cache / f'{key}.npz'
    top_p = cache / f'{key}_top.pkl'
    if not npz.exists() or not top_p.exists():
        return row_idx, np.nan, np.nan
    data = np.load(npz, allow_pickle=True)
    top = load_top(top_p)
    return row_idx, heavy_clash_count(data['xyz_gt'], top), heavy_clash_count(data['xyz_pred'], top)


def run_parallel(tasks, worker, n_workers, label):
    out = {}
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(worker, task): task for task in tasks}
        for fut in as_completed(futures):
            res = fut.result()
            out[res[0]] = res[1:]
            done += 1
            if done % 500 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed else 0
                eta = (len(tasks) - done) / rate if rate else 0
                print(f'[{label}] {done}/{len(tasks)}  {elapsed:.0f}s  {rate:.1f}/s  ETA {eta:.0f}s', flush=True)
    return out


def recompute_fig3(n_workers):
    out_dir = PROJECT / 'logs/figure3_step100'
    pf = out_dir / 'per_frame_metrics.csv'
    ps = out_dir / 'per_system_summary.csv'
    backup(pf)
    backup(ps)

    df = pd.read_csv(pf)
    print(f'[fig3] loaded {len(df)} rows; old clash mean={df["clash"].mean():.3f}', flush=True)
    tasks = [(i, r.ped_id, int(r.frame_idx)) for i, r in df.iterrows()]
    res = run_parallel(tasks, fig3_worker, n_workers, 'fig3')
    clash = np.full(len(df), np.nan)
    for i, vals in res.items():
        clash[i] = vals[0]
    df['clash'] = clash.astype(int)
    df.to_csv(pf, index=False)

    summary = pd.read_csv(ps)
    clash_summary = df.groupby('ped_id')['clash'].agg(['mean', 'std']).reset_index()
    clash_summary = clash_summary.rename(columns={'mean': 'clash_mean', 'std': 'clash_std'})
    summary = summary.drop(columns=[c for c in ['clash_mean', 'clash_std'] if c in summary.columns])
    summary = summary.merge(clash_summary, on='ped_id', how='left')
    summary.to_csv(ps, index=False)
    print(f'[fig3] new clash mean={df["clash"].mean():.3f}, median={df["clash"].median():.0f}, max={df["clash"].max()}', flush=True)
    return df, summary


def recompute_fig2(n_workers):
    out_dir = PROJECT / 'logs/figure2_step100'
    pc = out_dir / 'per_conformer_metrics.csv'
    ps = out_dir / 'per_system_summary.csv'
    panel = out_dir / 'figure2_panelF_clash.csv'
    backup(pc)
    backup(ps)
    backup(panel)

    df = pd.read_csv(pc)
    print(f'[fig2] loaded {len(df)} rows; old pred clash mean={df["clash_pred"].mean():.3f}', flush=True)
    tasks = [(i, r.ped_id, int(r.conformer_idx)) for i, r in df.iterrows()]
    res = run_parallel(tasks, fig2_worker, n_workers, 'fig2')
    clash_gt = np.full(len(df), np.nan)
    clash_pred = np.full(len(df), np.nan)
    for i, vals in res.items():
        clash_gt[i] = vals[0]
        clash_pred[i] = vals[1]

    old_gt = df['clash_gt'].copy()
    old_pred = df['clash_pred'].copy()
    df['clash_gt'] = clash_gt.astype(int)
    df['clash_pred'] = clash_pred.astype(int)
    df.to_csv(pc, index=False)

    panel_df = df[['ped_id', 'conformer_idx', 'clash_gt', 'clash_pred']].copy()
    panel_df.to_csv(panel, index=False)

    summary = pd.read_csv(ps)
    agg = df.groupby('ped_id').agg(
        clash_gt_mean=('clash_gt', 'mean'),
        clash_gt_std=('clash_gt', 'std'),
        clash_pred_mean=('clash_pred', 'mean'),
        clash_pred_std=('clash_pred', 'std'),
    ).reset_index()
    agg['clash_delta'] = agg['clash_pred_mean'] - agg['clash_gt_mean']
    agg['clash_rel_delta_pct'] = np.where(
        agg['clash_gt_mean'] != 0,
        agg['clash_delta'] / agg['clash_gt_mean'] * 100,
        np.nan,
    )
    clash_cols = ['clash_gt_mean', 'clash_gt_std', 'clash_pred_mean', 'clash_pred_std', 'clash_delta', 'clash_rel_delta_pct']
    summary = summary.drop(columns=[c for c in clash_cols if c in summary.columns])
    summary = summary.merge(agg[['ped_id'] + clash_cols], on='ped_id', how='left')
    summary.to_csv(ps, index=False)

    print(f'[fig2] new pred clash mean={df["clash_pred"].mean():.3f}; old/new max abs diff pred={np.max(np.abs(old_pred - df["clash_pred"]))}', flush=True)
    print(f'[fig2] new gt clash mean={df["clash_gt"].mean():.3f}; old/new max abs diff gt={np.max(np.abs(old_gt - df["clash_gt"]))}', flush=True)
    return df, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', choices=['fig2', 'fig3', 'both'], default='both')
    ap.add_argument('--workers', type=int, default=24)
    args = ap.parse_args()
    os.chdir(PROJECT)
    if args.target in ('fig3', 'both'):
        recompute_fig3(args.workers)
    if args.target in ('fig2', 'both'):
        recompute_fig2(args.workers)


if __name__ == '__main__':
    main()
