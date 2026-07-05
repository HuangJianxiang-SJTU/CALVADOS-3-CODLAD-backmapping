"""
Aim 1 — COCOMO2 single-chain IDP simulation (Feig lab COCOMO v2).

Mirrors run_openabc.py so COCOMO2 results are directly comparable to the
MOFF2/Mpipi length sweep: same SARW initial geometry (via init_sarw.py), same
per-block Rg equilibration gate, same clock-JSON output schema, same
--skip_prod_if_not_converged behavior. The only differences are the force
field (COCOMO2 v2 published defaults) and the env (cg_ensemble).

COCOMO2 is constructed with a bare OpenMM topology + positions (no Assembly),
so we can run single-chain IDPs without the condensate/assembly machinery.

Reference: https://github.com/feiglab/cocomo

Usage:
    python run_cocomo_singlechain.py --ped_id PED00016 --ca_pdb ... --gpu 0 \
        --equil_max_blocks 60 --skip_prod_if_not_converged
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import mdtraj as md
from openmm import app
import openmm.unit as unit

# init_sarw lives next door in the openabc dir; reuse it so initial geometry is
# identical across all three models (fair comparison).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "openabc"))
from init_sarw import build_sarw_ca_chain, stable_seed, write_pdb  # noqa: E402

from cocomo.cocomo_model import COCOMO  # noqa: E402


def build_topology_and_positions(ca_pdb):
    """Load a CA-only PDB as an OpenMM Topology + positions (nm)."""
    pdb = app.PDBFile(str(ca_pdb))
    top = pdb.topology
    pos = pdb.getPositions()
    return top, pos


def box_for(n_res):
    """Same CALVADOS heuristic the OpenABC sweep uses: (n-1)*0.38+4 nm.
    Keeps COCOMO2 single-chain in the same dilute-phase box regime as the
    MOFF2/Mpipi runs (no PBC self-interaction for max Rg ~6 nm)."""
    return int(np.ceil((n_res - 1) * 0.38 + 4))


def setup_sim(model, temp_k, timestep_fs, platform, gpu, positions=None):
    """Wrap COCOMO.setup_simulation and HARD-FAIL if CUDA was requested but
    COCOMO silently fell back to CPU (it catches OpenMMException and downgrades
    without warning). Per project policy: always GPU, never OpenCL/CPU."""
    model.setup_simulation(temperature=temp_k, gamma=0.01,
                           tstep=timestep_fs / 1000.0,  # ps
                           resources=platform, device=gpu,
                           positions=positions, resetvelocities=True)
    if platform == "CUDA":
        used = getattr(model, "resources", None)
        if used != "CUDA":
            raise RuntimeError(
                f"COCOMO fell back to '{used}' (CUDA requested) — aborting. "
                f"Check that the openmm-cuda plugin is installed in this env.")
        try:
            ctx_platform = model.simulation.context.getPlatform().getName()
        except Exception:
            ctx_platform = "?"
        if ctx_platform != "CUDA":
            raise RuntimeError(
                f"Context platform is '{ctx_platform}', not CUDA — aborting.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="cocomo2", help="output dir label")
    p.add_argument("--ped_id", required=True)
    p.add_argument("--ca_pdb", required=True, help="input CA pdb (topology source)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--replica", type=int, default=0)
    p.add_argument("--init", default="sarw", choices=["sarw", "straight"])
    p.add_argument("--sarw_clash", type=float, default=0.40)
    p.add_argument("--temp", type=float, default=295.0)
    p.add_argument("--timestep_fs", type=float, default=10.0)
    p.add_argument("--n_relax_steps", type=int, default=500)
    p.add_argument("--scramble_steps", type=int, default=5000)
    p.add_argument("--scramble_temp", type=float, default=450.0)
    p.add_argument("--equil_block", type=int, default=100000)
    p.add_argument("--equil_max_blocks", type=int, default=60,
                   help="cap on equilibration blocks (block=100k steps); 60 => 6M steps")
    p.add_argument("--equil_min_blocks", type=int, default=8)
    p.add_argument("--equil_slope_tol", type=float, default=0.01,
                   help="|slope of block-mean Rg| < tol*|mean Rg| => converged")
    p.add_argument("--n_production_steps", type=int, default=1000000)
    p.add_argument("--n_frames", type=int, default=1000)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--skip_prod_if_not_converged", action="store_true",
                   help="if equilibration does not converge by the block cap, skip "
                        "production and mark the replica non-converged (no DCD written)")
    p.add_argument("--box_nm", type=float, default=None,
                   help="cubic box edge (nm); default = (n-1)*0.38+4 (matches OpenABC sweep)")
    p.add_argument("--platform", default="CUDA", choices=["CUDA", "CPU"])
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-replica top_pdb: 4 reps of one PED share out_dir and run concurrently,
    # so a shared filename would race (same bug fixed in run_openabc.py).
    top_pdb = out_dir / f"{args.ped_id}_rep{args.replica}_first.pdb"
    seq_traj = md.load_pdb(args.ca_pdb)
    three2one = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
                 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
                 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}
    sequence = ''.join(three2one[r.name] for r in seq_traj.topology.residues)
    assert seq_traj.n_chains == 1, f"{args.ca_pdb}: expected 1 chain"

    if args.init == "sarw":
        seed = stable_seed(args.ped_id, args.replica)
        df = build_sarw_ca_chain(sequence, r0=0.38, clash_cutoff=args.sarw_clash,
                                 seed=seed)
        write_pdb(df, str(top_pdb))
        init_source = f"sarw(seed={seed})"
    else:
        import shutil
        shutil.copyfile(args.ca_pdb, top_pdb)
        init_source = "straight"

    # Build topology + positions from the (SARW or straight) CA pdb.
    top, pos = build_topology_and_positions(top_pdb)
    n_res = top.getNumResidues()
    box_nm = args.box_nm if args.box_nm is not None else box_for(n_res)

    input_parameters = {
        "model": args.model, "ped_id": args.ped_id, "replica": args.replica,
        "ca_pdb": str(args.ca_pdb), "init": args.init, "init_source": init_source,
        "temperature": args.temp, "ionic_strength_mM": None,
        "timestep_fs": args.timestep_fs,
        "n_relax_steps": args.n_relax_steps, "n_equil_steps": args.equil_block,
        "scramble_steps": args.scramble_steps, "scramble_temp": args.scramble_temp,
        "equil_block": args.equil_block, "equil_max_blocks": args.equil_max_blocks,
        "equil_slope_tol": args.equil_slope_tol, "equil_min_blocks": args.equil_min_blocks,
        "n_production_steps": args.n_production_steps,
        "n_frames": args.n_frames, "save_interval": args.save_interval,
        "platform": args.platform, "gpu": args.gpu,
        "cocomo_version": 2,
    }
    with open(out_dir / "input_parameters.json", "w") as f:
        json.dump(input_parameters, f, indent=4); f.write("\n")

    # COCOMO2 v2 with published defaults: version=2 => default params/eps/
    # surfscale; cuton=2.9, cutoff=3.1, kappa=1.0 (all __init__ defaults). Single
    # chain => no intercomp_repulsion, no ENM (IDP, disordered), no interactions.
    # NB: COCOMO._normalize_box treats a plain scalar as Angstrom, so pass the
    # box as a nanometer Quantity to avoid a 10x size error.
    model = COCOMO(topology=top, version=2, box=box_nm * unit.nanometer, positions=pos)

    # Setup simulation: Langevin, 295 K, 10 fs, CUDA mixed precision (COCOMO2
    # default). COCOMO2's terms differ from Mpipi's Wang-Frenkel contact; mixed
    # precision should be stable, but watch the first blocks for NaN.
    setup_sim(model, args.temp, args.timestep_fs, args.platform, args.gpu)

    # Relax: minimize + short warmup.
    model.minimize(nstep=500)
    model.set_velocities()
    model.simulate(nstep=args.n_relax_steps)

    # High-temperature scramble to erase SARW construction memory (same rationale
    # as run_openabc.py). COCOMO.setup_simulation rebuilds the context; carry
    # positions across and restore target T afterward.
    t_eq0 = time.perf_counter()
    if args.scramble_steps > 0:
        pos_scr = model.get_positions()
        setup_sim(model, args.scramble_temp, args.timestep_fs, args.platform,
                  args.gpu, positions=pos_scr)
        model.simulate(nstep=args.scramble_steps)

    # Restore target temperature for equilibration + production.
    pos_eq = model.get_positions()
    setup_sim(model, args.temp, args.timestep_fs, args.platform, args.gpu,
              positions=pos_eq)

    sim = model.simulation
    block_means = []
    converged = False
    blocks_run = 0
    while blocks_run < args.equil_max_blocks:
        sim.step(args.equil_block)
        blocks_run += 1
        pos_nm = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        rg = float(np.sqrt(((pos_nm - pos_nm.mean(0)) ** 2).sum(1).mean()))
        block_means.append(rg)
        print(f"  [equil] {args.ped_id} rep{args.replica} block {blocks_run}: "
              f"Rg={rg:.3f} nm", flush=True)
        if not np.isfinite(rg):
            print(f"  [equil] NaN detected at block {blocks_run}; aborting", flush=True)
            converged = False
            break
        if blocks_run >= max(args.equil_min_blocks, 4):
            recent = np.array(block_means[-4:])
            x = np.arange(len(recent))
            slope = np.polyfit(x, recent, 1)[0]
            mean_r = recent.mean()
            if abs(slope) < args.equil_slope_tol * abs(mean_r):
                converged = True
                print(f"  [equil] converged at block {blocks_run} "
                      f"(slope={slope:.4f}, mean Rg={mean_r:.3f})", flush=True)
                break
    eq_seconds = time.perf_counter() - t_eq0
    if not converged:
        print(f"  [equil] NOT converged after {blocks_run} blocks "
              f"(last block-mean Rg={block_means[-1] if block_means else 'n/a'})", flush=True)

    skip_production = (not converged) and args.skip_prod_if_not_converged
    if skip_production:
        print(f"  [equil] {args.ped_id} rep{args.replica} non-converged; "
              f"skipping production per --skip_prod_if_not_converged", flush=True)

    output_dcd = out_dir / f"{args.ped_id}_rep{args.replica}.dcd"
    wall_seconds = 0.0
    if not skip_production:
        model.simulate(nstep=args.n_production_steps, nout=args.save_interval,
                       dcdfile=str(output_dcd))
        wall_seconds = time.perf_counter()  # not precise, but recorded
        # COCOMO.simulate does not return wall time; approximate from block rate.
        wall_seconds = eq_seconds * (args.n_production_steps / (args.equil_block * max(blocks_run, 1)))

    clock = {
        **input_parameters, "box_nm": box_nm, "n_residues": n_res,
        "equil_seconds": eq_seconds,
        "wall_seconds_total": wall_seconds + eq_seconds,
        "equil_converged": converged,
        "production_run": not skip_production,
        "equil_blocks_run": blocks_run,
        "equil_block_mean_Rg_nm": block_means,
        "ns_per_day": (blocks_run * args.equil_block * args.timestep_fs / 1000.0)
                       / (eq_seconds / 86400.0) if eq_seconds > 0 else 0.0,
        "note": "COCOMO2 v2 published defaults; single-chain, no Assembly/ENM/repulsion",
    }
    with open(out_dir / f"clock_time_rep{args.replica}.json", "w") as f:
        json.dump(clock, f, indent=4); f.write("\n")

    print(f"[cocomo2 {args.ped_id} rep{args.replica}] done: "
          f"{blocks_run} equil blocks, Rg={block_means[-1] if block_means else 'n/a'}"
          + (" [NON-CONVERGED, no production]" if skip_production else ""))


if __name__ == "__main__":
    main()
