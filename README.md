# CALVADOS-3 / CODLAD Backmapping Pipeline

Reproducible figure generation for the manuscript *"Local Geometry Recovers but Cooperative Structure Does Not: Residue-Resolved Limits of All-Atom Reconstruction from Single-Bead Coarse-Grained Disordered Protein Ensembles"*.

This repository contains plotting scripts, data-processing scripts, intermediate CSVs, cached
backmapped conformations, and the final manuscript figure PDFs. All figures can be regenerated
from the provided inputs by running the processing and plotting scripts in the order described
below.

---

## Repository Structure

```
CALVADOS-3-CODLAD-backmapping/
├── README.md
├── scripts/                        # 5 plotting scripts → figures/
├── processing_scripts/             # 4 data-pipeline scripts → logs/
├── figures/                        # 8 final manuscript figure PDFs
├── dataset/                        # Cleaned system list
│   └── clean_v5.csv
└── logs/
    ├── figure2_step100/            # 4 input CSVs for Figure 2
    ├── figure3_step100/            # 3 input CSVs + ~25,000 cached NPZ/pkl frames
    ├── figure4/                    # 2 reference CSVs (step=5 legacy data)
    └── figure4_step100/            # 5 input CSVs for Figure 4
```

---

## File Inventory

### `scripts/` — Plotting Scripts

These scripts read processed CSVs from `logs/` and generate the figure PDFs in `figures/`.
Run them in any order after the data pipeline has completed.

| Script | Output | Description |
|--------|--------|-------------|
| `fig2_step100_plot.py` | `figure2.pdf`, `figure_s1.pdf` | All-atom reconstruction quality (2×2 main + 4×2 supp) |
| `fig3_step100_plot.py` | `figure3.pdf`, `figure_s2.pdf` | CG-conditioned reconstruction fidelity (2×2 main + 3×2 supp) |
| `fig4_step100_plot.py` | `figure4.pdf`, `figure_s4.pdf` | Per-residue helix recovery (5-stack main + 2×2 supp) |
| `figS3_rama_divergence_compute.py` | `figure_s3.pdf` | Ramachandran density divergence (3×6 grid) |
| `figure5.py` | `figure5_radar.pdf` | Three-condition radar summary |

### `processing_scripts/` — Data Pipeline Scripts

These scripts generate the CSV files consumed by the plotting scripts. Run them **in order**
(see Reproduction Workflow below).

| Script | Outputs | Depends on |
|--------|---------|------------|
| `fig2_step100_merge.py` | `logs/figure2_step100/*.csv` | `dataset/clean_v5.csv` |
| `fig3_step100_recompute_metrics.py` | `logs/figure3_step100/per_frame_metrics.csv` | `logs/figure3_step100/cache/*.npz` |
| `fig3_step100_merge.py` | `logs/figure3_step100/per_system_summary.csv`, `rama_three_way.csv`, `rama_by_class.csv`, `bond_geometry.csv`, `three_condition_decomposition.csv` | `dataset/clean_v5.csv`, `logs/figure3_step100/per_frame_metrics.csv`, `logs/figure2_step100/*.csv` |
| `fig4_step100_production.py` | `logs/figure4_step100/*.csv` | `logs/figure3_step100/cache/*.npz`, `logs/figure4/*.csv` |

### `dataset/` — System List

| File | Description |
|------|-------------|
| `clean_v5.csv` | Curated list of 23 PED systems used across all figures. Columns include PED identifier, protein name, sequence length, and validation flags. |

### `figures/` — Final Manuscript Figures

| File | Layout | Description |
|------|--------|-------------|
| `figure2.pdf` | 2×2 (A–D) | All-atom reconstruction quality: SC RMSD, Rg scatter, rotamer bars, clash counts |
| `figure3.pdf` | 2×2 (A–D) | CG-conditioned fidelity: Rg overlay, JS divergence, rotamer three-way, clash three-way |
| `figure4.pdf` | 5-stack (A1–A5) | Per-residue helix profiles across 5 systems with PED ground truth |
| `figure_s1.pdf` | 4×2 (A–H) | Figure 2 supplementary: Dmax, AA/BB/SC RMSD, SC/BB ratio, Cβ scatter, helix scatter, bond violins |
| `figure_s2.pdf` | 3×2 (A–F) | Figure 3 supplementary: bond violins, Cβ deviation, Dmax overlay, Pro φ, Trans-Pro, helix fraction |
| `figure_s3.pdf` | 3×6 grid | Ramachandran six-class, three-condition density divergence |
| `figure_s4.pdf` | 2×2 (A–D) | Figure 4 supplementary: Tif2 NRID zoom, ACTR zoom, p15PAF zoom, step comparison |
| `figure5_radar.pdf` | Radar | Three-condition aggregate summary (PED ref / PED+CODLAD / CG+CODLAD) |

### `logs/` — Intermediate Data and Cached Computations

#### `logs/figure2_step100/` — Figure 2 Inputs

| File | Rows | Description |
|------|------|-------------|
| `per_conformer_metrics.csv` | 4,156 | Per-conformer Rg, Dmax, RMSD, clash, Rama, rotamer metrics |
| `per_system_summary.csv` | 23 | Per-system mean metrics and standard deviations |
| `per_residue_type_sc_rmsd.csv` | 20 | Mean side-chain RMSD per amino acid type |
| `bond_geometry.csv` | 69 | Mean bond lengths (CA–C, N–CA, C–N) per system |

#### `logs/figure3_step100/` — Figure 3 Inputs & Cache

| File | Rows | Description |
|------|------|-------------|
| `per_frame_metrics.csv` | 12,500 | Per-CG-frame metrics (Rg, Dmax, Cα/Cβ deviation, bond geometry, Rama, rotamer, DSSP, clash) across 23 systems |
| `per_system_summary.csv` | 23 | Per-system means aggregated across CG frames |
| `rama_three_way.csv` | 6 | Ramachandran favoured percentages for three-condition comparison |
| `cache/*.npz` + `*_top.pkl` | ~25,000 files, 1.2 GB | Cached backmapped all-atom coordinates and topologies for every 8th CG frame. **Including these files allows skipping the expensive backmapping recomputation step.** |

#### `logs/figure4/` — Figure 4 Reference Data (Legacy Step=5)

| File | Description |
|------|-------------|
| `figure4_per_residue_helix_combined.csv` | Per-residue reference helix fractions from step=5 PED DSSP |
| `figure4_per_ref_helix_summary.csv` | Per-system summary of step=5 helix predictions |

#### `logs/figure4_step100/` — Figure 4 Inputs

| File | Rows | Description |
|------|------|-------------|
| `per_residue_helix.csv` | 533 | Per-residue DSSP helix fractions (step=100) with reference values |
| `per_system_summary.csv` | 5 | Per-system Pearson r, mean helix %, step comparison |
| `p15PAF_zoom.csv` | 7 | p15PAF residues 54–60 zoom data |
| `ACTR_zoom.csv` | 30 | ACTR N-terminal helix zoom data |
| `PED00184_zoom.csv` | 21 | Tif2 NRID residues 120–140 zoom data |

---


### Revision and Rebuttal Additions

The revised JCIM response adds reproducibility assets beyond the original five main/supplementary plotting scripts. New rebuttal figures and tables are stored under `manuscript/figures/` and `manuscript/tables/`, with generation scripts in `scripts/figures/`, `scripts/pulchra_backmapping/`, `scripts/aim1_cg_sampling/`, and `scripts/folded_pilot_benchmark.py`.

| Output | Script / Data Source | Purpose |
|--------|----------------------|---------|
| `manuscript/figures/experimental_method_stratification_rebuttal.*` | `scripts/figures/fig_experimental_method_stratification.py` | Provenance-stratified PED analysis for Supplementary Figure S1 |
| `manuscript/figures/pulchra_vs_codlad_rebuttal.*` | `scripts/figures/fig_pulchra_vs_codlad.py`; `logs/pulchra_{ped,cg}/per_system_summary.csv` | PULCHRA versus CODLAD reconstruction-method comparison for Supplementary Figure S6 |
| `manuscript/figures/phase_diagram_CG_ergodicity_rebuttal.*` | `scripts/aim1_cg_sampling/cocomo/plot_ergodicity_phase_diagram.py` | Alternative CG force-field sampling/ergodicity assessment for Supplementary Figure S7 |
| `manuscript/figures/folded_pilot_rebuttal.*` | `scripts/folded_pilot_benchmark.py`; `logs/folded_pilot_rebuttal/folded_pilot_per_condition_summary.csv` | Folded-domain pilot benchmark for Supplementary Figure S8 |
| `manuscript/figures/clash_length_compactness_rebuttal.*` | `scripts/figures/fig_clash_length_compactness_rebuttal.py` | Chain-length/compactness dependence of clash counts |

The PULCHRA source archive and source files used for the classical backmapping comparison are included under `src/pulchra/`. OpenABC/Mpipi/MOFF2 and COCOMO2 helper scripts used for the alternative-CG feasibility assessment are included under `scripts/aim1_cg_sampling/`. Large trajectory products and full PULCHRA per-frame caches are not committed; the committed summary CSVs are sufficient to regenerate the submitted rebuttal figures.

---

## Reproduction Workflow

All commands assume the repository root as the working directory. Scripts reference paths via a
`PROJECT` variable pointing to the repository root; update it inside each script if you clone
to a different location.

### Step 1: Generate Figure 2 intermediate data

```bash
python processing_scripts/fig2_step100_merge.py
```

**Inputs:** `dataset/clean_v5.csv`.
**Outputs:** `logs/figure2_step100/` — `per_conformer_metrics.csv`, `per_system_summary.csv`,
`per_residue_type_sc_rmsd.csv`, `bond_geometry.csv`.

### Step 2: Generate Figure 3 intermediate data

The cached NPZ frames in `logs/figure3_step100/cache/` provide pre-computed backmapped
coordinates. Use them to skip the expensive backmapping inference step:

```bash
python processing_scripts/fig3_step100_recompute_metrics.py
python processing_scripts/fig3_step100_merge.py
```

**Inputs:** `dataset/clean_v5.csv`, `logs/figure3_step100/cache/`, `logs/figure2_step100/`.
**Outputs:** `logs/figure3_step100/` — `per_frame_metrics.csv`, `per_system_summary.csv`,
`rama_three_way.csv`, `rama_by_class.csv`, `bond_geometry.csv`, `three_condition_decomposition.csv`.

### Step 3: Generate Figure 4 intermediate data

```bash
python processing_scripts/fig4_step100_production.py
```

**Inputs:** `logs/figure3_step100/cache/`, `logs/figure4/`.
**Outputs:** `logs/figure4_step100/` — `per_residue_helix.csv`, `per_system_summary.csv`,
`p15PAF_zoom.csv`, `ACTR_zoom.csv`, `PED00184_zoom.csv`.

### Step 4: Generate all figures

```bash
python scripts/fig2_step100_plot.py
python scripts/fig3_step100_plot.py
python scripts/fig4_step100_plot.py
python scripts/figS3_rama_divergence_compute.py
python scripts/figure5.py
```

**Outputs:** All 8 PDFs in `figures/`.

### Dependency Chain Summary

```
dataset/clean_v5.csv
    │
    ├─► processing_scripts/fig2_step100_merge.py ──► logs/figure2_step100/
    │
    ├─► logs/figure3_step100/cache/ (pre-computed, 25,000 files)
    │       │
    │       ├─► processing_scripts/fig3_step100_recompute_metrics.py
    │       │       └─► logs/figure3_step100/per_frame_metrics.csv
    │       │
    │       ├─► processing_scripts/fig3_step100_merge.py
    │       │       └─► logs/figure3_step100/ (summary CSVs)
    │       │
    │       └─► processing_scripts/fig4_step100_production.py
    │               └─► logs/figure4_step100/
    │
    └─► scripts/*.py ──► figures/*.pdf
```

---

## Dependencies

The scripts require Python 3.8+ with the following packages. Check individual script imports
for exact version requirements.

| Package | Used by | Purpose |
|---------|---------|---------|
| NumPy | all scripts | Numerical arrays |
| Pandas | all scripts | Tabular data I/O |
| Matplotlib | plotting scripts | Figure generation |
| SciPy | `fig3_step100_plot.py` | KS test |
| MDTraj | processing scripts | Trajectory I/O, DSSP, Ramachandran |
| OpenMM / PDBFixer | `fig4_step100_production.py` | Hydrogen placement for DSSP |

Install the core stack:

```bash
pip install numpy pandas matplotlib scipy mdtraj
```

OpenMM and PDBFixer are only required for `fig4_step100_production.py`; all other scripts
run without them.

---

## Citation

This repository accompanies the CALVADOS-3 / CODLAD backmapping manuscript. If you use this
code or data, please cite:

> *[Manuscript citation — DOI placeholder]*

---

## Contact

For questions, please open an issue on this repository or contact the corresponding author.
