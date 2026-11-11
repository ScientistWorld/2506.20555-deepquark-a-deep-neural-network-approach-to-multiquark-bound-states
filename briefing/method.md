# Method

The paper proposes DeepQuark, a deep-neural-network variational Monte Carlo method for multiquark bound states.

The Hamiltonian is the AL1 nonrelativistic constituent quark model: rest masses, kinetic energy, a color Coulomb term, a Gaussian-smeared hyperfine spin-spin term, and a linear confinement term. The method represents a color-singlet wavefunction in coupled color-spin-isospin bases instead of single-particle orbitals.

The neural wavefunction receives particle coordinates (center-of-mass frame), pairwise distances, and one-hot coupled-channel encodings. A fully connected tanh network (four hidden layers, about 1900 parameters for the tetraquarks and 2900 for the pentaquark) is multiplied by a Gaussian boundary factor, then projected with explicit identical-particle antisymmetrization and parity. Samples are drawn with Metropolis-Hastings moves combining Gaussian coordinate kicks and random discrete-channel flips. Parameters are optimized by minimizing the variational energy with stochastic reconfiguration; in the final stage ten checkpoints 50 steps apart are evaluated with a large sample and the lowest-energy one is kept.

This workspace vendors the authors' public `DeepQuark` code release under `method/DeepQuark` and adds only packaging, run-size settings, seeding hooks, a resumable final-stage continuation and an exporter (`multiquark/export.py`) that writes the selected state in the gym's submission format.

Reference schedule (`scripts/reproduce.sh`, released-code settings, `DQ_SEED=0`): 1000 SR iterations at lr 0.02 with 2e4 samples; blocks of 500 iterations with decaying lr until the local-energy variance check passes; ten final blocks of 50 iterations at lr 1e-3; selection of the lowest-energy final checkpoint by the method's own evaluation (2 x 1e5 samples each). For the pentaquark, whose final stage at lr 1e-3 hovers at the threshold, ten more final blocks at lr 1e-4 follow before the selection.
