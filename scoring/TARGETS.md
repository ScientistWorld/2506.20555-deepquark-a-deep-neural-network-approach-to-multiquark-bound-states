# Evaluation Targets

## Primary: tcc_al1_binding

Tests whether a solver finds a bound isoscalar vector doubly charmed tetraquark under the AL1 Hamiltonian.

Metric: `binding_mev`, lower is better.

Run evaluation with `bash scripts/evaluate.sh`.

## Primary: tbb_al1_binding

Tests whether a solver finds a deeply bound isoscalar vector doubly bottom tetraquark under the AL1 Hamiltonian.

Metric: `binding_mev`, lower is better.

Run evaluation with `bash scripts/evaluate.sh`.

## Partial / Optional: pcc_al1_binding

Tests whether a solver finds the weakly bound isoscalar spin-5/2 triply heavy pentaquark under the AL1 Hamiltonian.

Metric: `binding_mev`, lower is better.

The current reproduction artifacts include a partial Pccbar attempt, but it is not part of the accepted `core_claim_plus` score set because the reproduced binding remains too shallow relative to the paper. To score it explicitly, run `python eval/deepquark_scores.py --artifacts checkpoints --output scoring/scores.json --experiments pcc_al1_binding` and then inspect the result against `scoring/reference.json`.

## Constraints

All experiments also score `mc_error_mev`, `parameter_count`, and color-channel proportions. A scientist should improve binding-energy estimates without inflating the ansatz beyond the target architecture or collapsing the coupled-channel treatment. The reusable accepted evaluation currently emits Tcc and Tbb because those are the reproduced claims; Pccbar remains documented as the next target for future improvement.
