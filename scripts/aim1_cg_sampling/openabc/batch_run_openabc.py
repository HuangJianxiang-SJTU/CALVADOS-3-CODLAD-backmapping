"""
Batch launcher for OpenABC (Mpipi + MOFF2) single-chain IDP simulations.

Runs all 23 final PED systems × 4 replicas for each model, distributing work
across the 4 RTX 4090 GPUs (one process per GPU; replicas within a system run
sequentially on the assigned GPU). Mirrors batch_run_calvados.py's role.

Outputs (per model) to dataset/cg_simulations_{model}/{PED}/{PED}_rep{r}.dcd,
then merge_replicas.py combines the 4 per-replica DCDs into {PED}.dcd (4000
frames) + {PED}_first.pdb to match the CALVADOS layout that
scripts/fig3_step100_worker.py consumes.

Usage (openabc env):
    python batch_run_openabc.py --model moff2  [--gpus 0,1,2,3] [--dry-run]
    python batch_run_openabc.py --model mpipi
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT = '/MDdata/data04/jxhuang/cg_cascade'
OPENABC_PY = '/MDdata/data01/jxhuang/miniconda3/envs/openabc/bin/python'
RUNNER = os.path.join(PROJECT, 'scripts/aim1_cg_sampling/openabc/run_openabc.py')
CA_DIR = os.path.join(PROJECT, 'dataset/cg_input_ca')
SYSTEMS_CSV = os.path.join(PROJECT, 'dataset/clean_v5.csv')
N_REPLICAS = 4
N_FRAMES_PER_REPLICA = 1000      # 4 replicas -> 4000 frames
SAVE_INTERVAL = 1000             # steps between saved frames
N_RELAX_STEPS = 500
INIT = 'sarw'                    # SARW initial geometry (straight-chain start doesn't equilibrate)
SCRAMBLE_STEPS = 5000
SCRAMBLE_TEMP = 450.0
EQUIL_BLOCK = 100_000            # validated on PED00003: clean plateau with 100k blocks
EQUIL_MAX_BLOCKS = 20
EQUIL_MIN_BLOCKS = 8
EQUIL_SLOPE_TOL = 0.01
SKIP_PROD_IF_NOT_CONVERGED = False
TEMP = 295.0
IONIC_mM = 150.0


def load_systems():
    sys_list = []
    with open(SYSTEMS_CSV) as f:
        import csv
        for r in csv.DictReader(f):
            sys_list.append(r['ped_id'])
    return sys_list


def system_done(out_dir, ped_id, n_replicas):
    """Check all replica DCDs + clock files exist."""
    for r in range(n_replicas):
        dcd = Path(out_dir) / ped_id / f'{ped_id}_rep{r}.dcd'
        clk = Path(out_dir) / ped_id / f'clock_time_rep{r}.json'
        if not (dcd.exists() and clk.exists()):
            return False
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, choices=['mpipi', 'moff2'])
    p.add_argument('--gpus', default='0,1,2,3', help='Comma-separated GPU IDs')
    p.add_argument('--n_replicas', type=int, default=N_REPLICAS)
    p.add_argument('--n_frames', type=int, default=N_FRAMES_PER_REPLICA)
    p.add_argument('--save_interval', type=int, default=SAVE_INTERVAL)
    p.add_argument('--systems', default=None, help='Comma-separated PED IDs (default: all 23)')
    p.add_argument('--equil_max_blocks', type=int, default=EQUIL_MAX_BLOCKS,
                   help='cap on equilibration blocks (block=100k steps); 60 => 6M steps')
    p.add_argument('--skip_prod_if_not_converged', action='store_true',
                   help='skip production for replicas that do not pass the convergence gate')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    gpus = [int(g) for g in args.gpus.split(',')]
    systems = args.systems.split(',') if args.systems else load_systems()
    out_base = os.path.join(PROJECT, f'dataset/cg_simulations_{args.model}')

    # Build job list: (ped_id, replica) tuples, skipping completed ones.
    jobs = []
    for ped_id in systems:
        out_dir = os.path.join(out_base, ped_id)
        if system_done(out_base, ped_id, args.n_replicas):
            print(f'[skip] {ped_id}: all {args.n_replicas} replicas present')
            continue
        for r in range(args.n_replicas):
            rep_dcd = Path(out_dir) / f'{ped_id}_rep{r}.dcd'
            if rep_dcd.exists():
                continue
            jobs.append((ped_id, r))
    print(f'Model={args.model}  systems={len(systems)}  pending jobs={len(jobs)}  gpus={gpus}')

    if args.dry_run:
        for ped_id, r in jobs:
            print(f'  [dry] {ped_id} rep{r}')
        return

    # Round-robin jobs across GPUs, one process per GPU at a time.
    # Group jobs by GPU.
    t0 = time.time()
    procs = {}  # gpu -> (Popen, ped_id, rep)
    job_iter = iter(jobs)
    finished = 0

    def launch(gpu, ped_id, rep):
        out_dir = os.path.join(out_base, ped_id)
        os.makedirs(out_dir, exist_ok=True)
        cmd = [
            OPENABC_PY, RUNNER,
            '--model', args.model, '--ped_id', ped_id,
            '--ca_pdb', os.path.join(CA_DIR, ped_id, f'{ped_id}_ca.pdb'),
            '--out_dir', out_dir, '--gpu', str(gpu), '--replica', str(rep),
            '--n_frames', str(args.n_frames), '--save_interval', str(args.save_interval),
            '--n_relax_steps', str(N_RELAX_STEPS),
            '--init', INIT,
            '--scramble_steps', str(SCRAMBLE_STEPS), '--scramble_temp', str(SCRAMBLE_TEMP),
            '--equil_block', str(EQUIL_BLOCK), '--equil_max_blocks', str(args.equil_max_blocks),
            '--equil_min_blocks', str(EQUIL_MIN_BLOCKS), '--equil_slope_tol', str(EQUIL_SLOPE_TOL),
            '--temp', str(TEMP), '--ionic_mM', str(IONIC_mM),
            '--platform', 'CUDA',
        ]
        if args.skip_prod_if_not_converged:
            cmd.append('--skip_prod_if_not_converged')
        log = open(os.path.join(out_dir, f'rep{rep}.log'), 'w')
        return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), ped_id, rep

    # Fill all GPUs initially
    for gpu in gpus:
        try:
            ped_id, rep = next(job_iter)
            procs[gpu] = launch(gpu, ped_id, rep)
            print(f'[launch gpu{gpu}] {ped_id} rep{rep}')
        except StopIteration:
            break

    while procs:
        time.sleep(10)
        for gpu in list(procs.keys()):
            proc, ped_id, rep = procs[gpu]
            rc = proc.poll()
            if rc is not None:
                finished += 1
                elapsed = time.time() - t0
                status = 'OK' if rc == 0 else f'FAIL(rc={rc})'
                print(f'[done gpu{gpu}] {ped_id} rep{rep} {status}  ({finished}/{len(jobs)}, {elapsed/60:.1f}min)')
                # launch next on this GPU
                try:
                    npid, nrep = next(job_iter)
                    procs[gpu] = launch(gpu, npid, nrep)
                    print(f'[launch gpu{gpu}] {npid} rep{nrep}')
                except StopIteration:
                    del procs[gpu]

    print(f'All jobs done in {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
