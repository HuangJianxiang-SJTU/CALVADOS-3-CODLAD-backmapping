# Mpipi and MOFF2 OpenABC Tests

This folder contains the OpenABC-based scripts used to test Mpipi and MOFF2 single-chain IDP sampling for the JCIM revision.

Source package: OpenABC from the MIT/Zhang group GitHub repository, used through the installed `openabc` Python package. The scripts import `MpipiModel`, `MOFF2Model`, `MpipiProteinParser`, and `HPSParser` from OpenABC.

Relevant upstream project: https://github.com/ZhangGroup-MITChemistry/OpenABC

Scripts:

- `run_openabc.py`: runs one replica of one PED system with either `--model mpipi` or `--model moff2`.
- `batch_run_openabc.py`: launches multi-system, multi-replica Mpipi or MOFF2 jobs.
- `merge_replicas.py`: merges per-replica trajectories into the 4000-frame layout expected by the downstream analysis.
- `analyze_length_sweep.py`: summarizes late-window Rg behavior and replica dispersion.
- `init_sarw.py`: deterministic self-avoiding random-walk C-alpha initialization shared by the OpenABC and COCOMO2 tests.

These runs were used as CG-sampling feasibility tests, not as full cross-force-field reconstruction benchmarks. Downstream backmapping was not interpreted quantitatively unless replica-level CG convergence could first be established.
