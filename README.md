# CALVADOS-3 / CODLAD Backmapping Pipeline

This repository contains the scripts, processed summary tables, and submitted figures for the JCIM manuscript *Local Geometry Recovers but Cooperative Structure Does Not: Residue-Resolved Limits of All-Atom Reconstruction from Single-Bead Coarse-Grained Disordered Protein Ensembles*.

The repository is organized for figure-level reproducibility. Large raw trajectories and full per-frame reconstruction caches are not committed; the processed CSV files used to regenerate the submitted figures are included.

## What Is Included

- CODLAD reconstruction and metric scripts for PED-derived C-alpha inputs and CALVADOS-derived C-alpha trajectories.
- Plotting scripts for main-text Figures 2-5 and Supplementary Figures S1-S8.
- PULCHRA comparison scripts used for Supplementary Figure S6.
- Alternative CG force-field testing scripts for Mpipi, MOFF2, and COCOMO2 used for Supplementary Figure S7.
- Folded-domain pilot benchmark script used for Supplementary Figure S8.
- Processed manuscript tables under `manuscript/tables/`.

## Figure Reproduction Map

Main figures:

- Figure 2: `scripts/fig2_step100_plot.py` -> `figures/figure2.{pdf,png}`
- Figure 3: `scripts/fig3_step100_plot.py` -> `figures/figure3.{pdf,png}`
- Figure 4: `scripts/fig4_step100_plot.py` -> `figures/figure4.{pdf,png}`
- Figure 5: `scripts/figure5.py` -> `figures/figure5_radar.pdf`

Supplementary figures:

- Figure S1: `scripts/figures/fig_experimental_method_stratification.py` -> `manuscript/figures/figure_s1_experimental_method_stratification.*`
- Figure S2: `scripts/fig2_step100_plot.py` -> `figures/figure_s2.*`
- Figure S3: `scripts/fig3_step100_plot.py` -> `figures/figure_s3.*`
- Figure S4: `scripts/figS3_rama_divergence_compute.py` -> `figures/figure_s4.*`
- Figure S5: `scripts/fig4_step100_plot.py` -> `figures/figure_s5.*`
- Figure S6: `scripts/figures/fig_pulchra_vs_codlad.py` -> `manuscript/figures/figure_s6_pulchra_vs_codlad.*`
- Figure S7: `scripts/aim1_cg_sampling/cocomo/plot_ergodicity_phase_diagram.py` -> `manuscript/figures/figure_s7_cg_forcefield_ergodicity.*`
- Figure S8: `scripts/folded_pilot_benchmark.py --plot-only` -> `manuscript/figures/figure_s8_folded_pilot.*`

Response-only figure:

- Clash count versus length/compactness: `scripts/figures/fig_clash_length_compactness_rebuttal.py` -> `manuscript/figures/clash_length_compactness_rebuttal.*`

## Key Data Tables

- `manuscript/tables/ped_experimental_method_annotation.csv`: PED provenance annotations used for Supplementary Table S1.
- `manuscript/tables/experimental_method_stratification_summary.csv`: grouped metric summary used for Figure S1.
- `manuscript/tables/pulchra_vs_codlad_summary.csv`: PULCHRA/CODLAD comparison used for Figure S6.
- `manuscript/tables/cg_forcefield_ergodicity_summary_rebuttal.csv`: alternative-CG replica dispersion summary used for Figure S7.
- `manuscript/tables/folded_pilot_decomposition_rebuttal.csv`: folded-domain pilot summary used for Figure S8.

## Alternative CG Force-Field Tests

The revision includes feasibility tests of three additional CG force fields before attempting downstream backmapping comparisons.

- Mpipi and MOFF2 were run through OpenABC-based scripts in `scripts/aim1_cg_sampling/openabc/`.
- COCOMO2 was run through Feig-lab COCOMO scripts in `scripts/aim1_cg_sampling/cocomo/`.
- The force-field source repositories and implementation notes are documented in the README files inside those folders.
- Figure S7 is generated from the committed convergence summary table and lightweight clock JSON files, so large trajectories are not required to reproduce the submitted plot.

## Common Commands

Generate the main and shifted original supplementary figures:

```bash
python scripts/fig2_step100_plot.py
python scripts/fig3_step100_plot.py
python scripts/figS3_rama_divergence_compute.py
python scripts/fig4_step100_plot.py
python scripts/figure5.py
```

Generate the revision-added supplementary figures:

```bash
python scripts/figures/fig_experimental_method_stratification.py
python scripts/figures/fig_pulchra_vs_codlad.py
python scripts/aim1_cg_sampling/cocomo/plot_ergodicity_phase_diagram.py
python scripts/folded_pilot_benchmark.py --plot-only
```

Generate the response-only clash analysis:

```bash
python scripts/figures/fig_clash_length_compactness_rebuttal.py
```

## Dependencies

Plotting scripts require Python 3 with NumPy, Pandas, Matplotlib, and SciPy. Reconstruction and metric recomputation scripts additionally require MDTraj, OpenMM/PDBFixer, PyTorch, CODLAD, CALVADOS, and the relevant CG packages.

A lightweight OpenABC environment file is provided at `envs/openabc.yaml`. Full CODLAD/CALVADOS reconstruction requires the local model checkpoints and trajectory paths described in the manuscript methods and script headers.

## Notes on Large Files

The repository commits processed tables and final figure outputs. Full trajectories, full CODLAD reconstruction caches, and large per-frame PULCHRA caches are omitted because of size. The committed summary tables are sufficient to regenerate the submitted revision figures.

## Contact

Please open a GitHub issue or contact the corresponding author with questions about reproducing a figure or locating a processed input table.
