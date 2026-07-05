# CALVADOS-3 / CODLAD Backmapping Pipeline

This repository contains the analysis scripts, processed tables, cached reconstruction data, and final figure outputs for the manuscript *"Local Geometry Recovers but Cooperative Structure Does Not: Residue-Resolved Limits of All-Atom Reconstruction from Single-Bead Coarse-Grained Disordered Protein Ensembles"*.

The repository was updated for the JCIM revision so that figure filenames follow the final manuscript and Supplementary Information numbering. In particular, the new PED provenance analysis is Supplementary Figure S1; the original supplementary figures are therefore shifted to S2-S5; the reviewer-requested extension figures are S6-S8.

## Quick Figure Map

Use this table to locate each final figure, the script that generates it, and the main input tables. Files under `figures/` are the original manuscript figure set. Files under `manuscript/figures/` are revision/rebuttal additions.

| Final figure | Output file(s) | Generator script | Main inputs |
|---|---|---|---|
| Figure 2 | `figures/figure2.pdf` | `scripts/fig2_step100_plot.py` | `logs/figure2_step100/*.csv` |
| Figure 3 | `figures/figure3.pdf` | `scripts/fig3_step100_plot.py` | `logs/figure3_step100/*.csv` |
| Figure 4 | `figures/figure4.pdf` | `scripts/fig4_step100_plot.py` | `logs/figure4_step100/*.csv` |
| Figure 5 | `figures/figure5_radar.pdf` | `scripts/figure5.py` | hard-coded normalized values from the final summaries |
| Figure S1 | `manuscript/figures/figure_s1_experimental_method_stratification.{png,svg,pdf}` | `scripts/figures/fig_experimental_method_stratification.py` | `data/processed/cascade/ped_candidates/ped_full_inventory.csv`, `logs/figure2_step100/per_system_summary.csv`, `logs/figure3_step100/per_system_summary.csv` |
| Figure S2 | `figures/figure_s2.pdf` | `scripts/fig2_step100_plot.py` | `logs/figure2_step100/*.csv` |
| Figure S3 | `figures/figure_s3.pdf` | `scripts/fig3_step100_plot.py` | `logs/figure3_step100/*.csv` |
| Figure S4 | `figures/figure_s4.pdf` | `scripts/figS3_rama_divergence_compute.py` | Ramachandran point cache or the original Figure 2/Figure 3 reconstruction caches |
| Figure S5 | `figures/figure_s5.pdf` | `scripts/fig4_step100_plot.py` | `logs/figure4_step100/*.csv` |
| Figure S6 | `manuscript/figures/figure_s6_pulchra_vs_codlad.{png,svg,pdf}` | `scripts/figures/fig_pulchra_vs_codlad.py` | `logs/pulchra_ped/per_system_summary.csv`, `logs/pulchra_cg/per_system_summary.csv`, CODLAD summary CSVs |
| Figure S7 | `manuscript/figures/figure_s7_cg_forcefield_ergodicity.{png,svg,pdf}` | `scripts/aim1_cg_sampling/cocomo/plot_ergodicity_phase_diagram.py` | `manuscript/tables/cg_forcefield_ergodicity_summary_rebuttal.csv`, clock JSON files under `dataset/cg_simulations_{moff2,mpipi,cocomo2}/` |
| Figure S8 | `manuscript/figures/figure_s8_folded_pilot.{png,svg,pdf}` | `scripts/folded_pilot_benchmark.py` | `logs/folded_pilot_rebuttal/folded_pilot_per_condition_summary.csv` |
| Response-only clash analysis | `manuscript/figures/clash_length_compactness_rebuttal.{png,svg}` | `scripts/figures/fig_clash_length_compactness_rebuttal.py` | `manuscript/tables/clash_length_compactness_rebuttal.csv` |

## Table Map

| File | Used for | Description |
|---|---|---|
| `manuscript/tables/ped_experimental_method_annotation.csv` | Figure S1 / Table S1 | Per-system PED provenance annotation used in the revised SI dataset table. |
| `manuscript/tables/experimental_method_stratification_summary.csv` | Figure S1 | Grouped summary of reconstruction/reference metrics by PED provenance class. |
| `manuscript/tables/pulchra_vs_codlad_summary.csv` | Figure S6 | Summary metrics comparing CODLAD and PULCHRA under PED-derived and CALVADOS-derived C-alpha input. |
| `manuscript/tables/cg_forcefield_ergodicity_summary_rebuttal.csv` | Figure S7 | Per-replica late-window mean Rg values for MOFF2, Mpipi, COCOMO2, and CALVADOS 3. |
| `manuscript/tables/folded_pilot_decomposition_rebuttal.csv` | Figure S8 / response text | Folded-domain pilot decomposition values for Ubq2 and Gal3. |
| `manuscript/tables/clash_length_compactness_rebuttal.csv` | Response-only clash analysis | Per-system clash, length, and compactness values. |
| `manuscript/tables/clash_length_compactness_stats.csv` | Response-only clash analysis | Correlation statistics for clash count versus length/compactness. |

## Repository Structure

```text
CALVADOS-3-CODLAD-backmapping/
├── dataset/
│   ├── clean_v5.csv
│   └── cg_simulations_{moff2,mpipi,cocomo2}/   # lightweight clock JSONs for Figure S7
├── data/processed/cascade/ped_candidates/
│   └── ped_full_inventory.csv                  # PED inventory used for Figure S1 annotations
├── figures/                                    # final main figures and shifted SI Figures S2-S5
├── logs/                                       # processed metric tables and selected caches
├── manuscript/
│   ├── figures/                                # revision/rebuttal figures S1 and S6-S8
│   └── tables/                                 # revision/rebuttal summary tables
├── scripts/                                    # main figure and metric scripts
│   ├── aim1_cg_sampling/                       # alternative CG force-field scripts
│   ├── figures/                                # revision/rebuttal plotting scripts
│   └── pulchra_backmapping/                    # PULCHRA runner and metric scorer
└── src/pulchra/src_euplotes/                   # PULCHRA source used for the classical backmapper comparison
```

## Main Analysis Scripts

| Script | Role |
|---|---|
| `scripts/fig2_step100_worker.py` | CODLAD reconstruction from PED C-alpha traces. |
| `scripts/fig2_step100_merge.py` | Aggregates PED-to-CODLAD reconstruction metrics. |
| `scripts/fig2_step100_plot.py` | Generates Figure 2 and final Supplementary Figure S2. |
| `scripts/fig3_step100_worker.py` | CODLAD reconstruction from CALVADOS C-alpha trajectories. |
| `scripts/fig3_step100_recompute_metrics.py` | Recomputes structural metrics from cached CG-to-CODLAD frames. |
| `scripts/fig3_step100_recompute_clash_fast.py` | KDTree-based clash recomputation used to correct the CG-to-CODLAD clash column. |
| `scripts/fig3_step100_merge.py` | Aggregates CG-to-CODLAD metrics and three-condition summary tables. |
| `scripts/fig3_step100_plot.py` | Generates Figure 3 and final Supplementary Figure S3. |
| `scripts/figS3_rama_divergence_compute.py` | Generates final Supplementary Figure S4 and Ramachandran JS-divergence tables. |
| `scripts/fig4_step100_production.py` | Computes per-residue helix profiles. |
| `scripts/fig4_step100_plot.py` | Generates Figure 4 and final Supplementary Figure S5. |
| `scripts/figure5.py` | Generates Figure 5 radar summary. |

## Revision and Rebuttal Scripts

| Script | Output | Notes |
|---|---|---|
| `scripts/figures/fig_experimental_method_stratification.py` | Figure S1 and Table S1-related CSVs | Stratifies PED reference and reconstruction metrics by NMR-supported, SAXS-only, EPR/DEER, and idpGAN/CG-derived provenance. |
| `scripts/pulchra_backmapping/run_pulchra.py` | PULCHRA reconstruction caches | Runs PULCHRA on the same PED-derived and CALVADOS-derived C-alpha inputs used for CODLAD. |
| `scripts/pulchra_backmapping/recompute_metrics.py` | `logs/pulchra_{ped,cg}/per_system_summary.csv` | Scores PULCHRA outputs with the same metric definitions used for CODLAD. |
| `scripts/figures/fig_pulchra_vs_codlad.py` | Figure S6 | Compares CODLAD and PULCHRA across rotamers, C-beta deviation, clashes, helix fraction, Ramachandran favored percentage, and trans-Pro phi. |
| `scripts/aim1_cg_sampling/openabc/*.py` | Alternative-CG sampling helpers | Used for Mpipi and MOFF2 feasibility runs. |
| `scripts/aim1_cg_sampling/cocomo/*.py` | COCOMO2 sampling and Figure S7 plotting | Figure S7 uses committed summary values plus lightweight clock JSONs; large trajectories are not required. |
| `scripts/folded_pilot_benchmark.py` | Figure S8 and folded pilot tables | Runs or summarizes the Ubq2/Gal3 folded-domain contrast benchmark. |
| `scripts/figures/fig_clash_length_compactness_rebuttal.py` | Response-only clash analysis | Assesses how Figure 2D clash count depends on chain length and compactness. |

## Reproduction

All commands assume the repository root as the working directory. The processed CSVs and submitted figures are already committed, so readers can regenerate individual figures directly if the required Python environment is available.

Generate the main figures and shifted original supplementary figures:

```bash
python scripts/fig2_step100_plot.py
python scripts/fig3_step100_plot.py
python scripts/figS3_rama_divergence_compute.py
python scripts/fig4_step100_plot.py
python scripts/figure5.py
```

Figure S4 is a density-panel figure rather than a summary-table plot. Regenerating it from scratch requires either the original Figure 2/Figure 3 reconstruction caches or the precomputed Ramachandran point cache used during revision. The final submitted `figures/figure_s4.pdf` is committed directly.

Generate the revision/rebuttal figures:

```bash
python scripts/figures/fig_experimental_method_stratification.py
python scripts/figures/fig_pulchra_vs_codlad.py
python scripts/aim1_cg_sampling/cocomo/plot_ergodicity_phase_diagram.py
python scripts/folded_pilot_benchmark.py --plot-only
python scripts/figures/fig_clash_length_compactness_rebuttal.py
```

The full CODLAD and folded-pilot reconstruction workflows require the original local CODLAD/CALVADOS environments and checkpoint/data paths used in the study. For reproducibility of the submitted revision figures, the repository includes the processed summary CSVs used by the plotting scripts. `scripts/folded_pilot_benchmark.py --plot-only` regenerates Figure S8 from the committed folded-pilot summary table. Large trajectory products and full PULCHRA per-frame caches are intentionally not committed.

## Dependencies

The plotting scripts use Python 3 with NumPy, Pandas, Matplotlib, and SciPy. Metric recomputation and reconstruction scripts additionally use MDTraj, OpenMM/PDBFixer, PyTorch, CODLAD, CALVADOS, and model-specific CG packages as indicated by the script imports.

A lightweight OpenABC environment file used for the alternative-CG runs is provided at `envs/openabc.yaml`.

## Citation

This repository accompanies the CALVADOS-3 / CODLAD backmapping manuscript. If you use this code or data, please cite the manuscript once the final citation is available.

## Contact

For questions, please open an issue on this repository or contact the corresponding author.
