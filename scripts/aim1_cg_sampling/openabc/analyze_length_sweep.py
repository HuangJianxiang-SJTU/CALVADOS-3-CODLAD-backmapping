#!/usr/bin/env python
"""Analyze the OpenABC length-sweep: per-replica block-Rg trajectories and
classify each system x model as (a) clean, (b) basin-split, or (c) slow-mixing.

Reads clock_time_rep{R}.json from each cg_simulations_<model>/<PED>/ dir and
emits:
  - per-replica summary (converged, blocks run, first/last block Rg, CV spread)
  - the full block-mean Rg trajectory per replica (for trajectory inspection)
  - a classification per system x model with the reasoning shown

Usage:
  python analyze_length_sweep.py [--csv out.csv] [--traj traj.csv]
"""
import argparse, json, os
import numpy as np

SYSTEMS = ["PED00483", "PED00016", "PED00184", "PED00494"]
N_RES = {"PED00483": 53, "PED00016": 111, "PED00184": 154, "PED00494": 270}
MODELS = ["moff2", "mpipi"]
BASE = "dataset/cg_simulations_{}"


def load_rep(model, ped, rep):
    f = os.path.join(BASE.format(model), ped, f"clock_time_rep{rep}.json")
    if not os.path.exists(f):
        return None
    c = json.load(open(f))
    blk = np.array(c.get("equil_block_mean_Rg_nm", []), dtype=float)
    return {
        "rep": rep,
        "converged": bool(c["equil_converged"]),
        "production": bool(c.get("production_run", True)),
        "blocks": len(blk),
        "traj": blk,
        "final": float(blk[-1]) if len(blk) else float("nan"),
    }


def classify(reps):
    """Return (label, reasoning) where label in {a,b,c}.

    (a) clean: all reps converged AND across-rep final-Rg spread is small
              (max-min within ~10% of the mean, and each rep's late-window
               spread is small).
    (b) basin-split: reps individually flatten (low late-window slope/spread)
              but at DISTINCT final values -> non-interconverting basins.
    (c) slow-mixing: reps wander, late-window spread stays large within a rep,
              and/or across-rep trajectories cross but never settle.
    """
    finals = np.array([r["final"] for r in reps])
    mean_f, mn, mx = finals.mean(), finals.min(), finals.max()
    across = mx - mn
    rel_across = across / mean_f if mean_f > 0 else float("inf")

    # Per-rep late-window spread (last 8 blocks) — measures whether the rep
    # itself flattened.
    late_spreads = []
    late_slopes = []
    for r in reps:
        if len(r["traj"]) >= 8:
            late = r["traj"][-8:]
        else:
            late = r["traj"]
        late_spreads.append(late.max() - late.min())
        # abs slope of late window vs block index, normalized by mean
        if len(late) >= 2:
            x = np.arange(len(late))
            slope = abs(np.polyfit(x, late, 1)[0])
            late_slopes.append(slope / (late.mean() if late.mean() > 0 else 1.0))
        else:
            late_slopes.append(float("inf"))
    late_spreads = np.array(late_spreads)
    late_slopes = np.array(late_slopes)
    all_converged = all(r["converged"] for r in reps)
    flat_within = bool((late_slopes < 0.01).mean() >= 0.75)  # >=75% reps flattened

    reasoning = (
        f"finals={np.round(finals,2).tolist()} "
        f"mean={mean_f:.2f} across_range={across:.2f}nm ({rel_across*100:.0f}% of mean) "
        f"late_slope_rel={np.round(late_slopes,3).tolist()} "
        f"late_spread={np.round(late_spreads,2).tolist()} "
        f"all_converged={all_converged} flat_within={flat_within}"
    )

    if all_converged and rel_across < 0.10:
        return "a", reasoning
    if flat_within and rel_across >= 0.10:
        return "b", reasoning
    return "c", reasoning


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="logs/openabc_sweep/sweep_summary.csv")
    ap.add_argument("--traj", default="logs/openabc_sweep/sweep_trajectories.csv")
    args = ap.parse_args()

    rows = []
    traj_rows = []
    print("=" * 100)
    print("LENGTH-SWEEP ANALYSIS  (4 reps x 2 models x 4 systems; 6M-step equil cap)")
    print("=" * 100)
    for model in MODELS:
        print(f"\n### MODEL: {model}")
        for ped in SYSTEMS:
            reps = [load_rep(model, ped, r) for r in range(4)]
            reps = [r for r in reps if r is not None]
            if not reps:
                print(f"  {ped} (N={N_RES[ped]}): NO DATA")
                continue
            label, why = classify(reps)
            finals = [r["final"] for r in reps]
            mean_f = np.mean(finals)
            across = max(finals) - min(finals)
            nconv = sum(r["converged"] for r in reps)
            print(f"  {ped} (N={N_RES[ped]}): class=({label})  "
                  f"reps={len(reps)} conv={nconv}/{len(reps)}  "
                  f"final_Rg={np.round(finals,2).tolist()}  "
                  f"mean={mean_f:.2f} across={across:.2f}")
            print(f"      why: {why}")
            for r in reps:
                rows.append({
                    "model": model, "ped": ped, "N": N_RES[ped],
                    "rep": r["rep"], "converged": r["converged"],
                    "production": r["production"], "blocks": r["blocks"],
                    "final_Rg": round(r["final"], 3),
                    "mean_final_Rg": round(mean_f, 3),
                    "across_rep_range": round(across, 3),
                    "classification": label,
                })
                for i, v in enumerate(r["traj"]):
                    traj_rows.append({
                        "model": model, "ped": ped, "N": N_RES[ped],
                        "rep": r["rep"], "block": i + 1,
                        "block_Rg_nm": round(float(v), 4),
                    })

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    import csv
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(args.traj, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(traj_rows[0].keys()))
        w.writeheader()
        w.writerows(traj_rows)
    print(f"\nWrote {args.csv} ({len(rows)} rows) and {args.traj} ({len(traj_rows)} rows)")

    # Compact table
    print("\n" + "=" * 100)
    print("TABLE: N vs CV (across-rep Rg range) vs classification")
    print("=" * 100)
    print(f"{'model':<7}{'PED':<10}{'N':<5}{'class':<7}{'conv':<7}{'mean_Rg':<9}"
          f"{'across_range':<13}{'rel_range%':<11}")
    for model in MODELS:
        for ped in SYSTEMS:
            r = [x for x in rows if x["model"] == model and x["ped"] == ped]
            if not r:
                continue
            x = r[0]
            rel = x["across_rep_range"] / x["mean_final_Rg"] * 100 if x["mean_final_Rg"] else 0
            print(f"{model:<7}{ped:<10}{x['N']:<5}{x['classification']:<7}"
                  f"{sum(y['converged'] for y in r)}/{len(r):<4}"
                  f"{x['mean_final_Rg']:<9.2f}{x['across_rep_range']:<13.2f}{rel:<11.0f}")


if __name__ == "__main__":
    main()
