#!/usr/bin/env python3
"""Stratify PED reference/reconstruction metrics by deposited experimental provenance."""
from pathlib import Path
import textwrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT_FIG = ROOT / "manuscript" / "figures" / "figure_s1_experimental_method_stratification"
OUT_TABLE = ROOT / "manuscript" / "tables" / "experimental_method_stratification_summary.csv"
OUT_ANNOT = ROOT / "manuscript" / "tables" / "ped_experimental_method_annotation.csv"

PED_IDS = [
    "PED00003","PED00006","PED00016","PED00074","PED00154","PED00181","PED00184","PED00229","PED00231","PED00233","PED00238","PED00536",
    "PED00140","PED00141","PED00155","PED00183",
    "PED00493","PED00494","PED00495",
    "PED00442","PED00468","PED00483","PED00484",
]
GROUPS = {
    "PED00003": ("NMR-supported", "PED metadata: NMR/PRE"),
    "PED00006": ("NMR-supported", "PED metadata: NMR/PRE"),
    "PED00016": ("NMR-supported", "PED API: NMR chemical shifts, relaxation and RDC; SAXS flag also present"),
    "PED00074": ("NMR-supported", "PED metadata: NMR"),
    "PED00154": ("NMR-supported", "PED metadata: NMR chemical shifts/RDC"),
    "PED00181": ("NMR-supported", "PED metadata: NMR+SAXS; grouped by NMR-supported restraints"),
    "PED00184": ("NMR-supported", "PED metadata: NMR RDC"),
    "PED00229": ("NMR-supported", "PED metadata/title: CD, SAXS and NMR"),
    "PED00231": ("NMR-supported", "PED metadata/title: CD, SAXS and NMR"),
    "PED00233": ("NMR-supported", "PED metadata/title: CD, SAXS and NMR"),
    "PED00238": ("NMR-supported", "PED metadata/title: SAXS and NMR"),
    "PED00536": ("NMR-supported", "PED metadata: NMR/SAXS/PRE/RDC, reweighted MD"),
    "PED00140": ("SAXS-only", "PED API: SAXS experimental procedure"),
    "PED00141": ("SAXS-only", "PED API: SAXS experimental procedure"),
    "PED00155": ("SAXS-only", "PED metadata: SAXS"),
    "PED00183": ("SAXS-only", "PED API/title: SAXS-based ensemble"),
    "PED00493": ("EPR/DEER", "PED API: pulsed EPR/DEER experimental procedure"),
    "PED00494": ("EPR/DEER", "PED API: pulsed EPR/DEER experimental procedure"),
    "PED00495": ("EPR/DEER", "PED API: pulsed EPR/DEER experimental procedure"),
    "PED00442": ("idpGAN/CG-derived", "Official PED API: no experimental data integrated; idpGAN trained on COCOMO CG simulations; cg2all reconstruction and AMBER/OpenMM minimization; sequence from PED00203"),
    "PED00468": ("idpGAN/CG-derived", "Official PED API: no experimental data integrated; idpGAN trained on COCOMO CG simulations; cg2all reconstruction and AMBER/OpenMM minimization; sequence from PED00119"),
    "PED00483": ("idpGAN/CG-derived", "Official PED API: no experimental data integrated; idpGAN trained on COCOMO CG simulations; cg2all reconstruction and AMBER/OpenMM minimization; sequence from PED00011"),
    "PED00484": ("idpGAN/CG-derived", "Official PED API: no experimental data integrated; idpGAN trained on COCOMO CG simulations; cg2all reconstruction and AMBER/OpenMM minimization; sequence from PED00006"),
}
ORDER = ["NMR-supported", "SAXS-only", "EPR/DEER", "idpGAN/CG-derived"]
COLORS = {
    "NMR-supported": "#3B6FB6",
    "SAXS-only": "#D08A1F",
    "EPR/DEER": "#4C8C5A",
    "idpGAN/CG-derived": "#8E5A9E",
}

inv = pd.read_csv(ROOT / "data/processed/cascade/ped_candidates/ped_full_inventory.csv")
fig2 = pd.read_csv(ROOT / "logs/figure2_step100/per_system_summary.csv")
fig3 = pd.read_csv(ROOT / "logs/figure3_step100/per_system_summary.csv")

ann_rows = []
for pid in PED_IDS:
    row = inv.loc[inv.PED_ID == pid].iloc[0]
    group, note = GROUPS[pid]
    flags = []
    for col, label in [("has_nmr", "NMR"), ("has_saxs", "SAXS"), ("has_pre", "PRE"), ("has_chemshifts", "chemical shifts"), ("has_rdc", "RDC")]:
        if bool(row[col]):
            flags.append(label)
    ann_rows.append({
        "ped_id": pid,
        "experimental_group": group,
        "ped_metadata_flags": "; ".join(flags) if flags else "none in local PED inventory",
        "n_models": int(row.n_models),
        "title": row.title,
        "annotation_note": note,
    })
ann = pd.DataFrame(ann_rows)
ann.to_csv(OUT_ANNOT, index=False)

m = ann.merge(fig2, left_on="ped_id", right_on="ped_id", how="left").merge(
    fig3.add_prefix("cg_"), left_on="ped_id", right_on="cg_ped_id", how="left"
)
metrics = {
    "PED->CODLAD backbone RMSD (A)": "bb_rmsd_pred_mean",
    "PED->CODLAD side-chain RMSD (A)": "sc_rmsd_pred_mean",
    "PED reference clash count": "clash_gt_mean",
    "PED reference rotamer favored (%)": "rota_favored_pct_gt",
    "CG->CODLAD clash count": "cg_clash_mean",
    "CG->CODLAD rotamer favored (%)": "cg_rota_favored_pct",
}
summary_rows = []
for group in ORDER:
    sub = m[m.experimental_group == group]
    for label, col in metrics.items():
        vals = pd.to_numeric(sub[col], errors="coerce").dropna()
        summary_rows.append({
            "experimental_group": group,
            "n_systems": len(sub),
            "metric": label,
            "n_with_metric": len(vals),
            "mean": vals.mean() if len(vals) else np.nan,
            "std": vals.std(ddof=1) if len(vals) > 1 else np.nan,
            "median": vals.median() if len(vals) else np.nan,
            "min": vals.min() if len(vals) else np.nan,
            "max": vals.max() if len(vals) else np.nan,
        })
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_TABLE, index=False)

plot_metrics = [
    ("bb_rmsd_pred_mean", "PED->CODLAD\nbackbone RMSD (A)", "A"),
    ("sc_rmsd_pred_mean", "PED->CODLAD\nside-chain RMSD (A)", "B"),
    ("clash_gt_mean", "PED reference\nclash count", "C"),
    ("rota_favored_pct_gt", "PED reference\nrotamer favored (%)", "D"),
]
fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8), constrained_layout=True)
for ax, (col, ylabel, panel) in zip(axes.ravel(), plot_metrics):
    for i, group in enumerate(ORDER):
        vals = pd.to_numeric(m.loc[m.experimental_group == group, col], errors="coerce").dropna().values
        if len(vals) == 0:
            continue
        rng = np.random.default_rng(abs(hash((group, col))) % 2**32)
        x = i + rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(x, vals, s=42, color=COLORS[group], edgecolor="white", linewidth=0.7, zorder=3)
        ax.plot([i-0.18, i+0.18], [np.median(vals), np.median(vals)], color="black", lw=1.5, zorder=4)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([g.replace("-", "-") + f"\n(n={len(m[m.experimental_group==g])})" for g in ORDER], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.text(-0.12, 1.05, panel, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
fig.suptitle("Reconstruction and PED-reference metrics stratified by ensemble provenance", fontsize=12)
for ext in ["png", "svg", "pdf"]:
    fig.savefig(f"{OUT_FIG}.{ext}", dpi=300)
print(f"Wrote {OUT_ANNOT}")
print(f"Wrote {OUT_TABLE}")
print(f"Wrote {OUT_FIG}.png/.svg/.pdf")
print(summary.pivot(index="experimental_group", columns="metric", values="mean").round(3).to_string())
