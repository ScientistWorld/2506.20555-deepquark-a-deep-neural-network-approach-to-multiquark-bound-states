# Evaluation Targets

## Primary: tcc_al1_binding

Tests whether a solver finds a bound isoscalar vector doubly charmed tetraquark under the AL1 Hamiltonian.

Metric: `binding_mev`, lower is better.

Run evaluation with `bash scripts/evaluate.sh`.

## Primary: tbb_al1_binding

Tests whether a solver finds a deeply bound isoscalar vector doubly bottom tetraquark under the AL1 Hamiltonian.

Metric: `binding_mev`, lower is better.

Run evaluation with `bash scripts/evaluate.sh`.

## Primary: pcc_al1_binding

Tests whether a solver finds the weakly bound isoscalar spin-5/2 triply heavy pentaquark under the AL1 Hamiltonian.

Metric: `binding_mev`, lower is better.

Run evaluation with `bash scripts/evaluate.sh`.

## Constraints

All experiments also score `mc_error_mev`, `parameter_count`, and color-channel proportions. A scientist should improve binding-energy estimates without inflating the ansatz beyond the target architecture or collapsing the coupled-channel treatment.
