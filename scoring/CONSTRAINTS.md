# Constraints

The physics on the other side of the comparison is fixed by `eval/` and cannot be changed by a submission:

- the AL1 Hamiltonian (masses, Coulomb, constant, hyperfine and linear-confinement terms with their color and spin matrix elements);
- the physical sectors: `ccqq` and `bbqq` isoscalar J^P = 1+, `ccqqc` isoscalar J^P = 5/2-, color singlet, positive spatial parity, Fermi statistics for identical quarks (light pair with I = 0);
- the thresholds 3878, 10644 and 5718 MeV;
- the sampler, its seed, the number of chains and samples, the finite-difference kinetic energy and every acceptance check.

A submission is only a log-psi function. It is scored as failed (worst value of every scored metric) if it is outside the sector, non-deterministic, depends on how its inputs are batched, has side effects, exceeds 10000 stored float64-equivalent constants, does not let the sampler converge, produces non-finite local energies, reports an energy implausibly far below threshold, or does not finish within 30 minutes per system. Reported numbers in any file are ignored.

Scale: the reference networks hold 1920 (tetraquark) and 2948 (pentaquark) stored constants. The budget allows larger ansatzes up to 10000; much larger models, other Hamiltonians or other sectors are different experiments, not improvements on these.

`eval/` never imports `method/`; `eval/train/` and `eval/test/` share no code.
