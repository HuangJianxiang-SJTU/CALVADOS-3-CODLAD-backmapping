"""
Ramachandran distribution scoring against Lovell 2003 favored/allowed regions.

Vendors the precomputed Lovell 2003 grid as a module-level data structure.
Scores phi/psi angles per residue per frame, then averages over the ensemble.
"""

import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Lovell 2003 Ramachandran boundaries (MolProbity definitions)
# Stored as polygon vertices in (phi, psi) degrees for each region type.
#
# Source: Lovell et al. (2003), "Structure validation by Calpha geometry:
# phi, psi and Cbeta deviation", Proteins 50:437-450.
# Digital encoding from Richardson lab MolProbity source.
#
# Each region is a list of (phi_min, phi_max, psi_min, psi_max) boxes
# that approximate the allowed contours.
# ---------------------------------------------------------------------------

# General case (not glycine, not proline, not pre-proline)
_FAVORED_GENERAL = [
    # Core alpha-helix
    (-180, -20, -80, 20),
    # Extended beta
    (-180, 180, 60, 180),
    (-180, 180, -180, -60),
    # Left-handed alpha
    (20, 100, -80, 80),
]

# Allowed region is broader — pad each favored box by allowed margin
_ALLOWED_MARGIN_PHI = 30.0
_ALLOWED_MARGIN_PSI = 30.0

# Glycine: Lovell 2003 favored regions for glycine (no Cβ, broader φ/ψ range).
# Ref: Lovell et al. (2003), Proteins 50:437-450, Table 1 "Gly" column.
# Three regions covering ~97% of glycine residues in the Top500 dataset:
_FAVORED_GLY = [
    (-180, 0, -180, 40),     # right-handed, negative psi (helix/strand)
    (-180, 0, 40, 180),      # right-handed, positive psi (PPII/beta)
    (20, 100, -180, 40),     # left-handed alpha (mirror of general Lα)
]
# Glycine allowed margin: ±20° (tighter than general ±30° because favored
# regions are already broader). This is consistent with MolProbity's approach
# where glycine disallowed is a small fraction (~3%) of the map.
_GLY_ALLOWED_MARGIN_PHI = 20.0
_GLY_ALLOWED_MARGIN_PSI = 20.0

# Proline: restricted phi
_FAVORED_PRO = [
    (-95, -35, -80, 20),  # alpha-helix-like, phi restricted
    (-95, -35, 60, 180),  # extended-like
]

# Pre-proline (residue before proline): slightly different distribution
_FAVORED_PREPRO = [
    (-180, -20, -80, 20),
    (-180, 180, 60, 180),
    (-180, 180, -180, -60),
    (20, 120, -80, 80),
]

# Ile/Val (beta-branched): general without the left-handed alpha region.
# Beta-branching at Cβ (two non-H substituents) sterically disfavors
# positive phi conformations. Ref: Read et al. (2011) Structure 19:1395-1412.
_FAVORED_ILEVAL = [
    (-180, -20, -80, 20),     # Core alpha-helix
    (-180, 180, 60, 180),     # Extended beta (psi positive)
    (-180, 180, -180, -60),   # Extended beta (psi negative)
]


def _in_boxes(phi, psi, boxes):
    """Check if (phi, psi) falls inside any bounding box."""
    for phi_min, phi_max, psi_min, psi_max in boxes:
        if phi_min <= phi <= phi_max and psi_min <= psi <= psi_max:
            return True
    return False


def _compute_allowed_boxes(favored_boxes, phi_margin, psi_margin):
    """Expand favored boxes by margin to get allowed regions."""
    allowed = []
    for phi_min, phi_max, psi_min, psi_max in favored_boxes:
        allowed.append((
            phi_min - phi_margin,
            phi_max + phi_margin,
            psi_min - psi_margin,
            psi_max + psi_margin,
        ))
    return allowed


def _get_region_boxes(res_name, next_res_name):
    """Return (favored_boxes, allowed_boxes) for a given residue context."""
    if res_name == "GLY":
        favored = _FAVORED_GLY
        phi_margin = _GLY_ALLOWED_MARGIN_PHI
        psi_margin = _GLY_ALLOWED_MARGIN_PSI
    elif res_name == "PRO":
        favored = _FAVORED_PRO
        phi_margin = _ALLOWED_MARGIN_PHI
        psi_margin = _ALLOWED_MARGIN_PSI
    elif next_res_name == "PRO":
        favored = _FAVORED_PREPRO
        phi_margin = _ALLOWED_MARGIN_PHI
        psi_margin = _ALLOWED_MARGIN_PSI
    elif res_name in ("ILE", "VAL"):
        favored = _FAVORED_ILEVAL
        phi_margin = _ALLOWED_MARGIN_PHI
        psi_margin = _ALLOWED_MARGIN_PSI
    else:
        favored = _FAVORED_GENERAL
        phi_margin = _ALLOWED_MARGIN_PHI
        psi_margin = _ALLOWED_MARGIN_PSI

    allowed = _compute_allowed_boxes(favored, phi_margin, psi_margin)
    return favored, allowed


def compute_ramachandran_scores(
    coords: np.ndarray,
    topology,
) -> dict:
    """Score phi/psi angles against Lovell 2003 favored/allowed regions.

    Args:
        coords: (n_frames, n_atoms, 3) in Angstrom
        topology: mdtraj.Topology

    Returns:
        dict with keys:
            pct_favored: float — % of residues in favored regions (ensemble avg)
            pct_allowed: float — % in allowed (including favored)
            pct_outlier: float — % outside allowed
            per_residue_favored: (n_res,) % favored per residue
            per_residue_outlier: (n_res,) % outlier per residue
            worst_3_outliers: list[(res_idx, res_name, pct_outlier)] — worst 3 residues
    """
    import mdtraj as md

    n_frames = coords.shape[0]
    traj = md.Trajectory(coords, topology)
    # MDTraj returns (indices, angles) tuple; take angles
    phi_raw = md.compute_phi(traj)
    psi_raw = md.compute_psi(traj)
    phi_all = phi_raw[1] if isinstance(phi_raw, (tuple, list)) else phi_raw
    psi_all = psi_raw[1] if isinstance(psi_raw, (tuple, list)) else psi_raw
    # phi_all: (n_frames, n_phi), psi_all: (n_frames, n_psi)

    phi_deg = np.degrees(phi_all)
    psi_deg = np.degrees(psi_all)

    n_res = topology.n_residues
    res_names = [r.name for r in topology.residues]

    def _class_label(res_name, next_name):
        if res_name == "GLY":
            return "gly"
        elif res_name == "PRO":
            return "pro"
        elif next_name == "PRO":
            return "prepro"
        elif res_name in ("ILE", "VAL"):
            return "ileval"
        else:
            return "general"

    # Per-residue counts
    n_favored = np.zeros(n_res, dtype=np.int32)
    n_allowed = np.zeros(n_res, dtype=np.int32)
    n_total = np.zeros(n_res, dtype=np.int32)

    # Per-class counters
    class_names = ["gly", "pro", "prepro", "ileval", "general"]
    class_favored = {c: 0 for c in class_names}
    class_allowed = {c: 0 for c in class_names}
    class_total = {c: 0 for c in class_names}

    for k in range(n_frames):
        for ri in range(n_res):
            # MDTraj: phi[i] is phi of residue i+1, psi[i] is psi of residue i
            # phi valid for residues 1..n_res-1, psi valid for residues 0..n_res-2
            phi = phi_deg[k, ri - 1] if ri > 0 else np.nan
            psi = psi_deg[k, ri] if ri < n_res - 1 else np.nan

            if np.isnan(phi) or np.isnan(psi):
                continue

            res_name = res_names[ri]
            next_name = res_names[ri + 1] if ri + 1 < n_res else ""

            favored, allowed = _get_region_boxes(res_name, next_name)
            cl = _class_label(res_name, next_name)

            n_total[ri] += 1
            class_total[cl] += 1
            if _in_boxes(phi, psi, favored):
                n_favored[ri] += 1
                n_allowed[ri] += 1
                class_favored[cl] += 1
                class_allowed[cl] += 1
            elif _in_boxes(phi, psi, allowed):
                n_allowed[ri] += 1
                class_allowed[cl] += 1
            # else: outlier

    # Percentages
    pct_favored = np.full(n_res, np.nan, dtype=np.float64)
    pct_outlier = np.full(n_res, np.nan, dtype=np.float64)
    mask = n_total > 0
    pct_favored[mask] = 100.0 * n_favored[mask] / n_total[mask]
    pct_allowed_val = np.full(n_res, np.nan, dtype=np.float64)
    pct_allowed_val[mask] = 100.0 * n_allowed[mask] / n_total[mask]
    pct_outlier[mask] = 100.0 - pct_allowed_val[mask]

    # Ensemble averages (over residues with data)
    avg_favored = float(np.nanmean(pct_favored))
    avg_allowed = float(np.nanmean(pct_allowed_val))
    avg_outlier = float(np.nanmean(pct_outlier))

    # Worst 3 residues by outlier fraction
    res_outlier = [(ri, res_names[ri], float(pct_outlier[ri]))
                   for ri in range(n_res) if not np.isnan(pct_outlier[ri])]
    res_outlier.sort(key=lambda x: -x[2])
    worst_3 = res_outlier[:3]

    # Per-class breakdown
    class_breakdown = {}
    for cl in class_names:
        tot = class_total[cl]
        if tot > 0:
            class_breakdown[cl] = {
                "n_instances": tot,
                "pct_favored": round(100.0 * class_favored[cl] / tot, 2),
                "pct_allowed": round(100.0 * class_allowed[cl] / tot, 2),
                "pct_outlier": round(100.0 * (tot - class_allowed[cl]) / tot, 2),
            }

    return {
        "pct_favored": avg_favored,
        "pct_allowed": avg_allowed,
        "pct_outlier": avg_outlier,
        "per_residue_favored": pct_favored,
        "per_residue_outlier": pct_outlier,
        "worst_3_outliers": worst_3,
        "n_residues_scored": int(np.sum(mask)),
        "n_frames": n_frames,
        "class_breakdown": class_breakdown,
    }
