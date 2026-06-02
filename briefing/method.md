# Method

The paper proposes DeepQuark, a deep-neural-network variational Monte Carlo method for multiquark bound states.

The Hamiltonian is the AL1 nonrelativistic constituent quark model: rest masses, kinetic energy in the center-of-mass frame, a color Coulomb term, a Gaussian-smeared hyperfine spin-spin term, and a linear confinement term. The method represents a color-singlet wavefunction in coupled color-spin-isospin bases instead of single-particle orbitals.

The neural wavefunction receives particle coordinates, pairwise distances, and one-hot coupled-channel encodings. A fully connected tanh network is multiplied by a Gaussian boundary factor, then projected with explicit identical-particle antisymmetrization and parity. Samples are drawn with Metropolis-Hastings moves combining Gaussian coordinate kicks and random discrete-channel flips. Parameters are optimized by minimizing the variational energy with stochastic reconfiguration.

This workspace vendors the authors' public `DeepQuark` code release under `method/DeepQuark` and adds only packaging, configurable run sizes, and evaluation wrappers.
