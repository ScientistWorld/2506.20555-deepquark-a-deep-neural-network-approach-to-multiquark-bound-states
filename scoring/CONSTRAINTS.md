# Constraints

Future submissions must keep the AL1 Hamiltonian, flavor content, spin, isospin, parity sector, and threshold definitions fixed for each named experiment.

The evaluator records ansatz parameter count. Improvements that come only from making the model substantially larger should be reported as new experiments rather than replacing the reference method in the existing targets.

Evaluation code must remain independent of `method/`. Method code may write artifacts in the documented format, but `scripts/evaluate.sh` must score those artifacts without importing solver internals.
