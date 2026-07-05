"""
Aim 1 — COCOMO2 multi-chain / phase-separation simulation.

COCOMO uses a directory-based setup (prep_assembly → simulate).
This script wraps the two-step workflow:
  1. prep_assembly: builds the OpenMM system from PDB + component files
  2. simulate: runs the MD

Required files in --rundir:
    ca.pdb             — Cα structure (one or more chains)
    components         — chain counts (mdsim config format)
    component_types    — per-type parameters
    interactions       — (optional) inter-type interaction overrides

Usage (cg_ensemble env):
    # Step 1 — build system once
    python run_cocomo.py prep --rundir <dir> --box 100 [--surf 0.7] [--gpu 0]

    # Step 2 — run MD
    python run_cocomo.py simulate --rundir <dir> --n_steps 10000000 [--restart restart.xml]

See src/cocomo/scripts/ and cocomo docs for component file format.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _run(cmd, cwd):
    """Run a shell command in the given directory."""
    print(f'[run_cocomo] {" ".join(cmd)}  (cwd={cwd})')
    subprocess.run(cmd, cwd=cwd, check=True)


# ---------------------------------------------------------------------------
# prep subcommand
# ---------------------------------------------------------------------------

def cmd_prep(args):
    rundir = os.path.abspath(args.rundir)
    os.makedirs(rundir, exist_ok=True)

    prep_script = os.path.join(
        os.path.dirname(__file__),
        '../../../src/cocomo/src/cocomo/cli/prep_assembly.py'
    )

    cmd = [
        sys.executable, prep_script,
        '--pdb', os.path.join(rundir, 'ca.pdb'),
        '--box', args.box,
        '--surf', str(args.surf),
        '--resources', args.platform,
        '--device', str(args.gpu_id),
    ]
    if args.repulsion is not None:
        cmd += ['--repulsion', str(args.repulsion)]

    _run(cmd, cwd=rundir)
    print(f'Prep done. system.xml written to {rundir}/')


# ---------------------------------------------------------------------------
# simulate subcommand
# ---------------------------------------------------------------------------

def cmd_simulate(args):
    from cocomo.cocomo_model import COCOMO
    from cocomo.system_handling import Assembly
    from mdsim import PDBReader

    rundir = os.path.abspath(args.rundir)
    os.makedirs(rundir, exist_ok=True)

    # Load assembly from existing component files + structure
    pdb_path = os.path.join(rundir, 'ca.pdb')
    components   = os.path.join(rundir, 'components')
    ctypes_path  = os.path.join(rundir, 'component_types')
    if not os.path.exists(ctypes_path):
        ctypes_path = os.path.join(rundir, 'component_types_files')
    interactions = os.path.join(rundir, 'interactions')

    s = PDBReader(pdb_path)
    if os.path.exists(interactions):
        asm = Assembly(components, ctypes_path, structure=s, interactions=interactions)
    else:
        asm = Assembly(components, ctypes_path, structure=s)

    xml_path = os.path.join(rundir, 'system.xml')
    cocomo = COCOMO(asm, xml=xml_path, restart=args.restart)

    cocomo.setup_simulation(
        temperature=args.temp,
        resources=args.platform,
        device=str(args.gpu_id),
        restart=args.restart,
        tstep=0.01,
    )

    dcd_path = os.path.join(rundir, 'traj.dcd')
    log_path = os.path.join(rundir, 'energy.log')
    chk_path = os.path.join(rundir, 'restart.chk')

    cocomo.simulate(
        nstep=args.n_steps,
        nout=args.n_out,
        dcdfile=dcd_path,
        logfile=log_path,
        chkfile=chk_path,
    )
    print(f'Done. Trajectory: {dcd_path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description='COCOMO2 simulation wrapper')
    sub = p.add_subparsers(dest='cmd', required=True)

    # --- prep ---
    sp = sub.add_parser('prep', help='Build OpenMM system (run once per system)')
    sp.add_argument('--rundir',    required=True)
    sp.add_argument('--box',       default='100', help='Box nm: x or x:y:z')
    sp.add_argument('--surf',      type=float, default=0.7)
    sp.add_argument('--repulsion', type=float, default=None)
    sp.add_argument('--platform',  default='CUDA', choices=['CUDA', 'CPU'])
    sp.add_argument('--gpu_id',    type=int, default=0)
    sp.set_defaults(func=cmd_prep)

    # --- simulate ---
    ss = sub.add_parser('simulate', help='Run MD from prepared system')
    ss.add_argument('--rundir',    required=True)
    ss.add_argument('--n_steps',   type=int, default=10_000_000)
    ss.add_argument('--n_out',     type=int, default=10_000)
    ss.add_argument('--temp',      type=float, default=298.0)
    ss.add_argument('--platform',  default='CUDA', choices=['CUDA', 'CPU'])
    ss.add_argument('--gpu_id',    type=int, default=0)
    ss.add_argument('--restart',   default=None, help='Checkpoint or XML restart file')
    ss.set_defaults(func=cmd_simulate)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
