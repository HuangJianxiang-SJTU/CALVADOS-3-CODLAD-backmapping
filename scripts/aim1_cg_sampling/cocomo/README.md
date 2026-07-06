# COCOMO2 Tests

This folder contains the COCOMO2 scripts used for the alternative-CG force-field feasibility analysis in the JCIM revision.

Source package: Feig lab COCOMO repository.

Upstream project: https://github.com/feiglab/cocomo

Scripts:

- `run_cocomo_singlechain.py`: runs one COCOMO2 single-chain replica with the same initialization, equilibration-gate, and clock-JSON conventions used for Mpipi/MOFF2.
- `run_cocomo.py`: auxiliary COCOMO runner retained from the testing workflow.
- `plot_ergodicity_phase_diagram.py`: generates Supplementary Figure S7 from the committed CG force-field convergence summary and lightweight timing JSON files.

The COCOMO2 tests were used to assess whether alternative CG models reached replica-level convergence under the same practical single-chain protocol. They are reported as a sampling-feasibility analysis rather than a full reconstruction-accuracy benchmark across force fields.
