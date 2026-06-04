# Problem

Solve nonrelativistic few-body bound states in a constituent quark model for multiquark systems with fixed flavor, total spin, total isospin, parity, and color-singlet constraints.

The task is to estimate the ground-state energy of selected AL1 Hamiltonians and decide whether a state is bound relative to the lowest allowed two-hadron threshold. A valid solution must handle continuous particle coordinates and discrete color-spin-isospin coupled channels while respecting identical-particle exchange symmetry.

The accepted scored gym targets are:

- isoscalar vector doubly charmed tetraquark `ccqq`, spin-parity `1+`
- isoscalar vector doubly bottom tetraquark `bbqq`, spin-parity `1+`

The paper also reports pentaquark predictions, but those are not part of the accepted `scores.json` baseline for this gym. All energies are reported in MeV and binding energies are measured relative to the fixed AL1 model thresholds used by the evaluator.
