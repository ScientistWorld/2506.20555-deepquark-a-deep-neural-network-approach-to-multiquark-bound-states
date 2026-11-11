# Experiments

All three experiments ask the same scientific question for a different multiquark system under one fixed quark-model Hamiltonian: how low a variational energy can a method reach in the required physical sector, and does it establish a bound state below the lowest two-hadron threshold? The evaluator measures the energy of the submitted state itself, so a lower score means a better approximation of the true ground state, never a better report.

## tcc_al1_binding

The isoscalar 1+ doubly charmed tetraquark. Its bound state is expected to be shallow (about 15 MeV) and spatially extended, close to a two-meson configuration, which makes it sensitive to how well a state captures long-range structure and to Monte Carlo noise. It establishes whether a method resolves a weakly bound molecular-like state and by how much it improves on the established basis-expansion benchmark (14 MeV binding).

## tbb_al1_binding

The isoscalar 1+ doubly bottom tetraquark: a compact, deeply bound state (about 150 MeV). It tests the opposite regime with the same Hamiltonian, where a method must reach high absolute accuracy on a strongly bound state (benchmark: 152 MeV binding).

## pcc_al1_binding

The isoscalar 5/2- triply heavy pentaquark: a five-body system with three coupled color channels, predicted to be bound by only a few MeV below its meson-baryon threshold. Basis-expansion methods struggle at five bodies, and the reference reaches only a shallow binding here, so this experiment measures how far a method can push a hard, larger system toward and past the predicted binding.

In every experiment the secondary scored metric is the Monte Carlo error of the measured energy at a fixed sample count: a state closer to an eigenstate has a smaller local-energy variance and a more precise energy.
