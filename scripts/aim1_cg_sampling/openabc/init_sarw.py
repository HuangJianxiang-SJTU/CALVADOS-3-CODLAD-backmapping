"""
Self-avoiding random walk (SARW) Cα chain generator for OpenABC single-chain
IDP simulations.

Replaces the straight-chain start (build_straight_CA_chain), which leaves the
chain fully extended (Rg ~ N^1) and diffusion-limits its collapse under the CG
force field. A SARW starts near equilibrium (Rg ~ N^0.5-0.6), so equilibration
is fast and not biased by a maximally-extended initial state.

Produces a Cα PDB with the same atom/residue layout as build_straight_CA_chain
(only the x/y/z columns differ), so it drops into the OpenABC parsers unchanged.

Usage:
    from init_sarw import build_sarw_ca_chain, write_sarw_pdb
    df = build_sarw_ca_chain(sequence, seed=42)
    write_sarw_pdb(df, 'PED00003_init.pdb')
"""
import hashlib
import numpy as np
import pandas as pd
from openabc.utils import build_straight_CA_chain, write_pdb


def stable_seed(ped_id, replica):
    """Deterministic 32-bit seed from (ped_id, replica) so each replica gets a
    different but reproducible SARW (independent of PYTHONHASHSEED)."""
    h = hashlib.sha256(f'{ped_id}#{replica}'.encode()).digest()
    return int.from_bytes(h[:4], 'big')


def sarw_coords(n, r0=0.38, clash_cutoff=0.40, rng=None, max_attempts=2000):
    """Self-avoiding random walk in nm.

    Bond length r0; rejects a candidate if it comes within clash_cutoff of any
    NON-adjacent already-placed bead (the immediately preceding bead is at r0
    by construction and is excluded). On failure to place after max_attempts
    (rare for N<=500), places the candidate anyway — minimizeEnergy + the
    high-T scramble resolve any residual local clash.
    """
    if rng is None:
        rng = np.random.default_rng()
    coords = np.zeros((n, 3))
    for i in range(1, n):
        placed = False
        for _ in range(max_attempts):
            v = rng.normal(size=3)
            v /= np.linalg.norm(v) + 1e-12
            cand = coords[i - 1] + r0 * v
            # exclude the bonded predecessor (i-1) from the clash check
            if i == 1:
                coords[i] = cand
                placed = True
                break
            d2 = ((coords[:i - 1] - cand) ** 2).sum(axis=1)
            if d2.min() > clash_cutoff ** 2:
                coords[i] = cand
                placed = True
                break
        if not placed:
            coords[i] = cand  # accept; minimize/scramble will relax
    coords -= coords.mean(axis=0)
    return coords


def build_sarw_ca_chain(sequence, r0=0.38, clash_cutoff=0.40, seed=0,
                        max_attempts=2000):
    """Return a CA-atom DataFrame (same schema as build_straight_CA_chain) with
    SARW coordinates (angstroms)."""
    df = build_straight_CA_chain(sequence, r0=r0)  # straight, correct schema
    rng = np.random.default_rng(seed)
    coords_nm = sarw_coords(len(sequence), r0=r0, clash_cutoff=clash_cutoff,
                            rng=rng, max_attempts=max_attempts)
    df['x'] = (coords_nm[:, 0] * 10).round(3)  # nm -> angstrom
    df['y'] = (coords_nm[:, 1] * 10).round(3)
    df['z'] = (coords_nm[:, 2] * 10).round(3)
    return df


def write_sarw_pdb(sequence, out_pdb, r0=0.38, clash_cutoff=0.40, seed=0,
                   max_attempts=2000):
    df = build_sarw_ca_chain(sequence, r0=r0, clash_cutoff=clash_cutoff,
                             seed=seed, max_attempts=max_attempts)
    write_pdb(df, out_pdb)
    return out_pdb
