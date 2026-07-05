"""
Generate the CG force-field ergodicity rebuttal figure.

Panel A: replica dispersion D(N) for MOFF2, Mpipi, COCOMO2, and CALVADOS 3.
Panel B-C: representative block-mean Rg traces showing MOFF2 basin splitting
and Mpipi slow mixing.
Panel D: CALVADOS 3 replica-segment Rg means from manuscript DCD trajectories.

OpenABC/COCOMO2 data are read from per-replica clock_time_rep{R}.json files.
CALVADOS 3 data are read from merged 4000-frame manuscript DCDs; the launcher
used 20 replicas with 200 kept frames each, so the trajectory is split into
20 contiguous replica segments.
"""
import csv
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import mdtraj as md
except Exception:
    md = None

OPENABC_SYSTEMS = [
    (53, "PED00483"),
    (111, "PED00016"),
    (154, "PED00184"),
    (270, "PED00494"),
    (457, "PED00183"),
]
COCOMO_SYSTEMS = [
    (111, "PED00016"),
    (270, "PED00493"),
    (457, "PED00183"),
]
CALVADOS_SYSTEMS = OPENABC_SYSTEMS
NS = [53, 111, 154, 270, 457]

MODEL_STYLE = {
    "MOFF2": ("o", "#d62728"),
    "Mpipi": ("^", "#1f77b4"),
    "COCOMO2": ("s", "#2ca02c"),
    "CALVADOS 3": ("D", "#222222"),
}

SUMMARY_TABLE = Path("manuscript/tables/cg_forcefield_ergodicity_summary_rebuttal.csv")


def clock_path(model, ped, rep):
    return Path(f"dataset/cg_simulations_{model}/{ped}/clock_time_rep{rep}.json")


def load_clock_blocks(model, ped, maxrep):
    blocks = []
    meta = []
    for rep in range(maxrep):
        f = clock_path(model, ped, rep)
        if not f.exists():
            continue
        with f.open() as handle:
            c = json.load(handle)
        b = np.asarray(c["equil_block_mean_Rg_nm"], dtype=float)
        blocks.append(b)
        meta.append(c)
    return blocks, meta


def late_means_from_clock(model, ped, maxrep):
    blocks, _ = load_clock_blocks(model, ped, maxrep)
    vals = []
    for b in blocks:
        late = b[len(b)//2:] if len(b) >= 4 else b
        vals.append(float(np.mean(late)))
    return vals


def calvados_replica_means(ped, n_rep=20):
    if md is None:
        raise RuntimeError("mdtraj is required to compute CALVADOS 3 Rg from DCD files")
    dcd = Path(f"dataset/cg_simulations/{ped}/{ped}.dcd")
    top = Path(f"dataset/cg_simulations/{ped}/{ped}_first.pdb")
    traj = md.load(str(dcd), top=str(top))
    rg = md.compute_rg(traj)
    if traj.n_frames % n_rep != 0:
        raise ValueError(f"{ped}: {traj.n_frames} frames cannot be split into {n_rep} replicas")
    seg = traj.n_frames // n_rep
    return [float(rg[i*seg:(i+1)*seg].mean()) for i in range(n_rep)]


def dispersion(vals):
    a = np.asarray(vals, dtype=float)
    if len(a) < 2:
        return np.nan
    return float(a.std(ddof=1) / a.mean())


def bootstrap_se_dispersion(vals, n_boot=20000, seed=12345):
    a = np.asarray(vals, dtype=float)
    if len(a) < 2:
        return np.nan
    rng = np.random.default_rng(seed + len(a) + int(round(a.mean() * 1000)))
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    samples = a[idx]
    boot = samples.std(axis=1, ddof=1) / samples.mean(axis=1)
    return float(np.nanstd(boot, ddof=1))


def build_data():
    data = {"MOFF2": {}, "Mpipi": {}, "COCOMO2": {}, "CALVADOS 3": {}}
    if SUMMARY_TABLE.exists():
        with SUMMARY_TABLE.open(newline="") as handle:
            for row in csv.DictReader(handle):
                model = row["model"]
                n = int(row["N"])
                data.setdefault(model, {}).setdefault(n, []).append(float(row["late_mean_rg_nm"]))
        return data

    for n, ped in OPENABC_SYSTEMS:
        maxrep = 8 if ped in {"PED00016", "PED00184"} else 4
        data["MOFF2"][n] = late_means_from_clock("moff2", ped, maxrep)
        data["Mpipi"][n] = late_means_from_clock("mpipi", ped, 4)
    for n, ped in COCOMO_SYSTEMS:
        data["COCOMO2"][n] = late_means_from_clock("cocomo2", ped, 4)
    for n, ped in CALVADOS_SYSTEMS:
        data["CALVADOS 3"][n] = calvados_replica_means(ped, n_rep=20)
    return data


def plot_regime_background(ax):
    ax.axhspan(0.00, 0.05, facecolor="#e8f5e9", alpha=0.7, zorder=0)
    ax.axhspan(0.05, 0.15, facecolor="#fff8e1", alpha=0.7, zorder=0)
    ax.axhspan(0.15, 0.25, facecolor="#ffebee", alpha=0.7, zorder=0)
    ax.axhline(0.05, color="0.55", lw=0.7, ls="--", zorder=1)
    ax.axhline(0.15, color="0.55", lw=0.7, ls="--", zorder=1)


def plot_phase(ax, data):
    plot_regime_background(ax)
    for model in ["MOFF2", "Mpipi", "COCOMO2", "CALVADOS 3"]:
        marker, color = MODEL_STYLE[model]
        xs, ys, ses = [], [], []
        for n in NS:
            vals = data[model].get(n, [])
            if len(vals) >= 2:
                xs.append(n)
                ys.append(dispersion(vals))
                ses.append(bootstrap_se_dispersion(vals))
        ls = "--" if model == "CALVADOS 3" else "-"
        ax.errorbar(xs, ys, yerr=ses, marker=marker, ms=5.8, lw=1.25,
                    capsize=2.8, color=color, markerfacecolor=color,
                    markeredgecolor=color, linestyle=ls, label=model, zorder=5)
    label_box = dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.2)
    ax.text(470, 0.025, "D < 0.05", ha="right", va="center", fontsize=7.5,
            color="0.35", bbox=label_box)
    ax.text(470, 0.10, "slow-mixing risk", ha="right", va="center", fontsize=7.5,
            color="0.35", bbox=label_box)
    ax.text(470, 0.20, "non-ergodic / split", ha="right", va="center", fontsize=7.5,
            color="0.35", bbox=label_box)
    ax.set_xlim(40, 480)
    ax.set_ylim(0.0, 0.25)
    ax.set_xticks(NS)
    ax.set_xlabel("Chain length N")
    ax.set_ylabel(r"Replica dispersion $D=\sigma(\langle R_g\rangle)/\langle\langle R_g\rangle\rangle$")
    ax.set_title("A  Replica dispersion across force fields", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")


def plot_clock_traces(ax, model, ped, maxrep, title, color, note=None):
    blocks, _ = load_clock_blocks(model, ped, maxrep)
    max_len = max(len(b) for b in blocks)
    for i, b in enumerate(blocks):
        x = np.arange(1, len(b) + 1)
        ax.plot(x, b, lw=1.15, alpha=0.85, color=color)
        ax.scatter(x[-1], b[-1], s=12, color=color, zorder=4)
    ax.axvspan(max_len//2 + 0.5, max_len + 0.5, color="0.85", alpha=0.35, lw=0)
    ax.set_xlabel("Equilibration block (100k steps)")
    ax.set_ylabel(r"Block mean $R_g$ (nm)")
    ax.set_title(title, loc="left", fontweight="bold")
    if note:
        ax.text(0.03, 0.97, note, transform=ax.transAxes, va="top", ha="left",
                fontsize=7.5, color="0.25",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5))


def plot_calvados_panel(ax, data):
    marker, color = MODEL_STYLE["CALVADOS 3"]
    xs = []
    for n in NS:
        vals = data["CALVADOS 3"].get(n, [])
        if not vals:
            continue
        vals = np.asarray(vals, dtype=float)
        jitter = np.linspace(-5, 5, len(vals))
        ax.scatter(np.full(len(vals), n) + jitter, vals, s=13, color=color,
                   alpha=0.45, edgecolors="none")
        mean = vals.mean()
        ax.plot([n-8, n+8], [mean, mean], color="#d62728", lw=1.4)
        d = dispersion(vals)
        ax.text(n, mean + 0.16, f"D={d:.3f}", ha="center", va="bottom",
                fontsize=7.2, color="0.25")
        xs.append(n)
    ax.set_xlim(40, 480)
    ax.set_xticks(NS)
    ax.set_xlabel("Chain length N")
    ax.set_ylabel(r"CALVADOS 3 replica-segment mean $R_g$ (nm)")
    ax.set_title("D  CALVADOS 3 merged 20-replica trajectories", loc="left", fontweight="bold")
    ax.text(0.03, 0.97, "4000 frames split into 20 x 200-frame replica segments",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.5, color="0.25",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5))


def main():
    data = build_data()

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.linewidth": 0.9,
        "axes.edgecolor": "0.2",
        "figure.dpi": 100,
    })

    fig, axs = plt.subplots(2, 2, figsize=(9.2, 6.8))
    plot_phase(axs[0, 0], data)
    plot_clock_traces(
        axs[0, 1], "moff2", "PED00184", 8,
        "B  MOFF2 basin splitting example (N=154)", "#d62728",
        note="8 replicas; shaded region used for late-window means",
    )
    plot_clock_traces(
        axs[1, 0], "mpipi", "PED00183", 4,
        "C  Mpipi slow-mixing example (N=457)", "#1f77b4",
        note="4 replicas; continued late-window wandering",
    )
    plot_calvados_panel(axs[1, 1], data)

    for ax in axs.ravel():
        ax.tick_params(axis="both", labelsize=8, length=3.5, width=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("CG force-field sampling feasibility for cross-model backmapping", y=0.995,
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.975])

    out_png = "manuscript/figures/phase_diagram_CG_ergodicity_rebuttal.png"
    out_svg = "manuscript/figures/phase_diagram_CG_ergodicity_rebuttal.svg"
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, bbox_inches="tight", facecolor="white")
    print(f"saved {out_png} and {out_svg}")

    print(f"\n{'model':<11}{'N':<6}{'nrep':<6}{'D(N)':<8}{'SE_D':<8}")
    for model in ["MOFF2", "Mpipi", "COCOMO2", "CALVADOS 3"]:
        for n in NS:
            vals = data[model].get(n, [])
            if len(vals) >= 2:
                print(f"{model:<11}{n:<6}{len(vals):<6}{dispersion(vals):<8.3f}{bootstrap_se_dispersion(vals):<8.3f}")


if __name__ == "__main__":
    main()
