# Research Direction

Improve variational solutions of the fixed AL1 multiquark Hamiltonians: reach lower energies (deeper, better-resolved binding) with more precise estimates, in the given physical sectors, for a shallow molecular-like tetraquark, a compact tetraquark and a five-body pentaquark.

Acceptable directions include better wavefunction ansatzes (neural or explicit few-body forms such as correlated Gaussians) that respect the required symmetries, better optimization (natural-gradient variants, sample-size schedules, learning-rate schedules, longer training), better sampling and variance reduction during training, initialization and transfer between systems, and more reliable selection among trained states. The largest known gap is the pentaquark, where the paper reports about 3 MeV of binding and the reference reaches about 0.6 MeV.

Not acceptable: anything that does not submit a genuine wavefunction of the fixed problem, e.g. states outside the sector, functions that behave differently depending on how they are called, or attempts to influence the evaluator's sampling or Hamiltonian. The evaluator measures energies itself, so reported numbers have no effect.
