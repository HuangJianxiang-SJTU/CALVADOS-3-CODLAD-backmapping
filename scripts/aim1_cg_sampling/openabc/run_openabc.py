"""
Aim 1 — OpenABC single-chain IDP simulation for Mpipi or MOFF2.

Mirrors the role of calvados/run_idr.py but builds the system with OpenABC's
MpipiModel / MOFF2Model (both single-bead Cα), following the published
tutorials (tutorials/MOFF2-simulations/run_test_IDP.py and the Mpipi API).

Produces one replica DCD (Cα, nm) + topology, with wall-clock timing recorded,
to be merged with 3 sibling replicas into a 4000-frame trajectory matching the
CALVADOS layout that scripts/fig3_step100_worker.py consumes.

Usage (cg_ensemble env):
    python run_openabc.py --model moff2 --ped_id PED00074 \
        --ca_pdb dataset/cg_input_ca/PED00074/PED00074_ca.pdb \
        --out_dir dataset/cg_simulations_moff2/PED00074 --gpu 0 --replica 0

Conditions match CALVADOS: 295 K, 0.15 M ionic strength, single chain,
constraint-free Cα (IDP path — no elastic network / ordered-domain restraints).
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import mdtraj as md

try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
except ImportError:
    import simtk.openmm as mm
    import simtk.openmm.app as app
    import simtk.unit as unit

from openabc.forcefields.parsers import HPSParser, MpipiProteinParser
from openabc.forcefields.MOFF2.forcefields import MOFF2Model
from openabc.forcefields.mpipi_model import MpipiModel
from openabc.utils import write_pdb

from init_sarw import build_sarw_ca_chain, stable_seed


def build_model(model_name, ca_pdb, temperature_K, ionic_strength_mM):
    """Build the OpenABC CG system following the published IDP tutorial recipe."""
    if model_name == "moff2":
        model = MOFF2Model()
        model.append_mol(HPSParser(str(ca_pdb)))
        # Tutorial overrides (tutorials/MOFF2-simulations/run_test_IDP.py)
        model.protein_bonds.loc[:, "k_bond"] = 8000.0
        model.protein_bonds.loc[:, "r0"] = 0.386
    elif model_name == "mpipi":
        model = MpipiModel()
        model.append_mol(MpipiProteinParser(str(ca_pdb)))
        # Tutorial overrides for Mpipi protein bonds (same recipe as MOFF2 IDP)
        model.protein_bonds.loc[:, "k_bond"] = 8000.0
        model.protein_bonds.loc[:, "r0"] = 0.386
    else:
        raise ValueError(f"unknown model: {model_name}")

    # Single-chain sanity
    ca_traj = md.load_pdb(str(ca_pdb))
    assert ca_traj.n_chains == 1, f"{ca_pdb}: expected 1 chain, got {ca_traj.n_chains}"

    # HIS charge + charged termini (tutorial recipe); HIS dict charge already
    # set by the parser, but we enforce 0.5 for MOFF2 / keep parser value for
    # Mpipi, then adjust termini identically.
    charges = model.atoms["charge"].to_numpy().astype(float)
    if model_name == "moff2":
        his_mask = model.atoms["resname"] == "HIS"
        charges[his_mask] = 0.5
    charges[0] += 1.0
    charges[-1] -= 1.0
    model.atoms["charge"] = charges

    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["mpipi", "moff2"])
    p.add_argument("--ped_id", required=True)
    p.add_argument("--ca_pdb", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--replica", type=int, default=0)
    p.add_argument("--temp", type=float, default=295.0)
    p.add_argument("--ionic_mM", type=float, default=150.0)
    p.add_argument("--timestep_fs", type=float, default=10.0)
    p.add_argument("--box_nm", type=float, default=None,
                   help="cubic box edge (nm); default = CALVADOS heuristic (n-1)*0.38+4. "
                        "Use 1000 to match the OpenABC MOFF2-simulations tutorial (the model's "
                        "intended dilute-phase box for its density term).")
    p.add_argument("--n_frames", type=int, default=1000,
                   help="frames to save per replica (4 replicas -> 4000 total)")
    p.add_argument("--save_interval", type=int, default=1000,
                   help="steps between saved frames")
    p.add_argument("--n_relax_steps", type=int, default=500)
    p.add_argument("--n_equil_steps", type=int, default=100_000,
                   help="unsaved equilibration steps before production (straight-chain start needs this)")
    # --- SARW initial geometry (replaces pathological straight-chain start) ---
    p.add_argument("--init", default="sarw", choices=["sarw", "straight"],
                   help="initial Cα geometry: self-avoiding random walk (recommended) or straight chain")
    p.add_argument("--sarw_clash", type=float, default=0.40,
                   help="SARW clash cutoff (nm) for rejecting non-adjacent bead overlaps")
    p.add_argument("--scramble_steps", type=int, default=5000,
                   help="high-T scramble steps after minimize to erase SARW construction memory")
    p.add_argument("--scramble_temp", type=float, default=450.0,
                   help="temperature (K) for the scramble phase")
    # --- Block-averaging convergence gate for equilibration ---
    # Validated on PED00003 (MOFF2): 100k-step blocks with an 8-block slope
    # window give a clean plateau (block-means ~3.0-3.9nm, no trend); smaller
    # 50k blocks / 4-block window fired on a local plateau and missed a slow
    # drift. Keep these defaults unless re-validating.
    p.add_argument("--equil_block", type=int, default=100_000,
                   help="equilibration block size (steps) over which to average Rg")
    p.add_argument("--equil_max_blocks", type=int, default=20,
                   help="cap on equilibration blocks; production starts once converged or at cap")
    p.add_argument("--equil_slope_tol", type=float, default=0.01,
                   help="|slope of block-mean Rg vs block index| < tol * |mean Rg| => converged")
    p.add_argument("--equil_min_blocks", type=int, default=8,
                   help="minimum blocks before the slope test is applied")
    p.add_argument("--skip_prod_if_not_converged", action="store_true",
                   help="if equilibration does not converge by the block cap, skip the "
                        "production run and mark the replica non-converged (no DCD written)")
    p.add_argument("--platform", default="CUDA", choices=["CUDA", "OpenCL", "CPU"])
    args = p.parse_args()

    n_production_steps = args.n_frames * args.save_interval
    ca_pdb = Path(args.ca_pdb)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Topology + (optionally) initial coordinates come from the input CA pdb.
    # The straight-chain inputs in dataset/cg_input_ca/ are valid topologies;
    # for --init sarw we regenerate coordinates as a self-avoiding random walk
    # (deterministic per replica) so the chain starts compact instead of fully
    # extended — MOFF2/Mpipi collapse from an extended start is diffusion-limited
    # and does not equilibrate on a practical timescale.
    # Per-replica filename: 4 reps of one PED share out_dir and run concurrently,
    # so a shared "{ped_id}_first.pdb" races (one rep's build_model can read
    # another's half-written SARW file -> parse_pdb sees no ATOM lines -> KeyError).
    top_pdb = out_dir / f"{args.ped_id}_rep{args.replica}_first.pdb"
    seq_traj = md.load_pdb(str(ca_pdb))
    three2one = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
                 'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
                 'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
    sequence = ''.join(three2one[r.name] for r in seq_traj.topology.residues)
    assert seq_traj.n_chains == 1, f"{ca_pdb}: expected 1 chain"

    if args.init == "sarw":
        seed = stable_seed(args.ped_id, args.replica)
        df = build_sarw_ca_chain(sequence, r0=0.38, clash_cutoff=args.sarw_clash,
                                 seed=seed)
        write_pdb(df, str(top_pdb))
        init_source = f"sarw(seed={seed})"
    else:
        shutil.copyfile(ca_pdb, top_pdb)
        init_source = "straight"

    input_parameters = {
        "model": args.model, "ped_id": args.ped_id, "replica": args.replica,
        "ca_pdb": str(ca_pdb), "init": args.init, "init_source": init_source,
        "temperature": args.temp, "ionic_strength_mM": args.ionic_mM,
        "timestep_fs": args.timestep_fs,
        "n_relax_steps": args.n_relax_steps, "n_equil_steps": args.n_equil_steps,
        "scramble_steps": args.scramble_steps, "scramble_temp": args.scramble_temp,
        "equil_block": args.equil_block, "equil_max_blocks": args.equil_max_blocks,
        "equil_slope_tol": args.equil_slope_tol, "equil_min_blocks": args.equil_min_blocks,
        "n_production_steps": n_production_steps,
        "n_frames": args.n_frames, "save_interval": args.save_interval,
        "platform": args.platform, "gpu": args.gpu,
    }
    with open(out_dir / "input_parameters.json", "w") as f:
        json.dump(input_parameters, f, indent=4); f.write("\n")

    # Build system from the (SARW or straight) CA pdb.
    model = build_model(args.model, top_pdb, args.temp, args.ionic_mM)
    top = app.PDBFile(str(top_pdb)).getTopology()

    # Box: the OpenABC MOFF2/Mpipi density term is parameterized for a large
    # (slab/condensate) box; the tutorial uses 1000 nm. The CALVADOS heuristic
    # (n-1)*0.38+4 nm is far smaller and puts the density term outside its
    # intended regime — likely the cause of slow non-converging drift.
    n_res = len(model.atoms.index)
    if args.box_nm is not None:
        box_nm = args.box_nm
    else:
        box_nm = int(np.ceil((n_res - 1) * 0.38 + 4))
    model.create_system(top=top, box_a=box_nm, box_b=box_nm, box_c=box_nm)

    if args.model == "moff2":
        model.add_protein_bonds(force_group=1)
        model.add_moff2_forces(
            temperature=args.temp,
            ionic_strength=args.ionic_mM,
            res_group_mapping="default",
            contact_force_group=2, elec_force_group=3,
            density_force_group_start=4,
        )
    else:  # mpipi
        model.add_protein_bonds(force_group=1)
        model.add_contacts(force_group=2)
        model.add_dh_elec(force_group=3)

    with open(out_dir / "system.xml", "w") as f:
        f.write(mm.XmlSerializer.serialize(model.system))

    # Integrator + platform. Double precision on CUDA: Mpipi's Wang-Frenkel
    # contact term goes NaN under mixed precision on this GPU. CG systems are
    # tiny, so the cost is negligible. Do NOT set CUDA_VISIBLE_DEVICES here:
    # we select the device via DeviceIndex, and remasking invalidates that index.
    T = args.temp * unit.kelvin
    integrator = mm.LangevinMiddleIntegrator(T, 1.0 / unit.picosecond,
                                              args.timestep_fs * unit.femtosecond)
    if args.platform == "CUDA":
        properties = {"Precision": "double", "DeviceIndex": str(args.gpu)}
    else:
        properties = {}
    init_coord = app.PDBFile(str(top_pdb)).getPositions()
    model.set_simulation(integrator, platform_name=args.platform,
                         init_coord=init_coord, properties=properties)

    # Relax: minimize + short warmup at the target temperature.
    model.simulation.minimizeEnergy()
    model.simulation.context.setVelocitiesToTemperature(T)
    model.simulation.step(args.n_relax_steps)

    # High-temperature scramble to erase residual memory of the SARW construction
    # (local clashes / directional bias). We only care about the equilibrium
    # ensemble, so kinetics here are irrelevant. Swap the integrator to T_scramble,
    # run, then restore the target-temperature integrator for equilibration.
    t_eq0 = time.perf_counter()
    if args.scramble_steps > 0:
        T_scr = args.scramble_temp * unit.kelvin
        scr_integ = mm.LangevinMiddleIntegrator(T_scr, 1.0 / unit.picosecond,
                                                args.timestep_fs * unit.femtosecond)
        # Recreate context at scramble temperature (set_simulation rebuilds the
        # context with the new integrator; carry current positions).
        pos = model.simulation.context.getState(getPositions=True).getPositions()
        model.set_simulation(scr_integ, platform_name=args.platform,
                             init_coord=pos, properties=properties)
        model.simulation.context.setVelocitiesToTemperature(T_scr)
        model.simulation.step(args.scramble_steps)

    # Equilibration with a convergence gate: run in blocks, compute mean Rg per
    # block, and declare equilibrated once the block-mean Rg slope (normalized
    # by the mean) falls below equil_slope_tol over the last few blocks.
    # Restore target-temperature integrator for equilibration + production.
    pos = model.simulation.context.getState(getPositions=True).getPositions()
    integrator = mm.LangevinMiddleIntegrator(T, 1.0 / unit.picosecond,
                                              args.timestep_fs * unit.femtosecond)
    model.set_simulation(integrator, platform_name=args.platform,
                         init_coord=pos, properties=properties)
    model.simulation.context.setVelocitiesToTemperature(T)

    sim = model.simulation
    top_mdtraj = md.load_topology(str(top_pdb))
    block_means = []
    converged = False
    blocks_run = 0
    while blocks_run < args.equil_max_blocks:
        sim.step(args.equil_block)
        blocks_run += 1
        pos_nm = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        # Rg of a single bead-per-residue chain: bead masses are residue masses,
        # but for a uniform-bead approximation Rg_unweighted is what we need to
        # track convergence (slope), not the absolute value.
        rg = float(np.sqrt(((pos_nm - pos_nm.mean(0)) ** 2).sum(1).mean()))
        block_means.append(rg)
        print(f"  [equil] {args.ped_id} rep{args.replica} block {blocks_run}: "
              f"Rg={rg:.3f} nm", flush=True)
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
              f"(last block-mean Rg={block_means[-1]:.3f})", flush=True)

    skip_production = (not converged) and args.skip_prod_if_not_converged
    if skip_production:
        print(f"  [equil] {args.ped_id} rep{args.replica} non-converged; "
              f"skipping production per --skip_prod_if_not_converged", flush=True)

    output_dcd = out_dir / f"{args.ped_id}_rep{args.replica}.dcd"
    if skip_production:
        wall_seconds = 0.0
    else:
        model.add_reporters(report_interval=args.save_interval, output_dcd=str(output_dcd))

        # Production with wall-clock timing
        t0 = time.perf_counter()
        model.simulation.step(n_production_steps)
        wall_seconds = time.perf_counter() - t0

        # Final state + checkpoint
        checkpoint_path = out_dir / f"{args.ped_id}_rep{args.replica}.chk"
        model.simulation.saveCheckpoint(str(checkpoint_path))
        final_pdb = out_dir / f"{args.ped_id}_rep{args.replica}_final.pdb"
        state = model.simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
        with open(final_pdb, "w") as f:
            app.PDBFile.writeFile(model.simulation.topology, state.getPositions(), f)

    clock = {
        **input_parameters, "box_nm": box_nm, "n_residues": n_res,
        "wall_seconds": wall_seconds,
        "equil_seconds": eq_seconds,
        "wall_seconds_total": wall_seconds + eq_seconds,
        "equil_converged": converged,
        "production_run": not skip_production,
        "equil_blocks_run": blocks_run,
        "equil_block_mean_Rg_nm": block_means,
        "ns_per_day": (n_production_steps * args.timestep_fs / 1000.0)
                       / (wall_seconds / 86400.0) if wall_seconds > 0 else None,
    }
    with open(out_dir / f"clock_time_rep{args.replica}.json", "w") as f:
        json.dump(clock, f, indent=4); f.write("\n")

    print(f"[{args.model} {args.ped_id} rep{args.replica}] done: "
          f"{args.n_frames} frames, {wall_seconds:.1f}s -> {output_dcd}"
          + (" [NON-CONVERGED, no production]" if skip_production else ""))


if __name__ == "__main__":
    main()
