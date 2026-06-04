# Constraints

Future submissions must keep the AL1 Hamiltonian and the named physical sectors fixed:

- `tcc_al1_binding`: isoscalar vector `ccqq` tetraquark with the 3878 MeV two-meson threshold used by the evaluator.
- `tbb_al1_binding`: isoscalar vector `bbqq` tetraquark with the 10644 MeV two-meson threshold used by the evaluator.

The reproduced reference artifacts use 1857 scored parameters for both accepted tetraquark targets. Future improvements should use comparable model scale for these experiments; if a scientist substantially increases the parameter count, changes the Hamiltonian, changes the threshold, or changes the flavor/spin/isospin sector, that result should be reported as a new experiment rather than replacing the fixed target.

Scoring artifacts must preserve the evaluator-visible fields: total energy in MeV, binding energy in MeV, Monte Carlo error in MeV, parameter count, and color-channel proportions. A method that improves binding while inflating noise, omitting the color-channel accounting, or relying on a different threshold should score worse on the constraint metrics.

Evaluation code must remain independent of `method/`. Method code may write artifacts in the documented format under `checkpoints/`, but `scripts/evaluate.sh` must score those artifacts without importing solver internals.
