# Problem

Find the ground states of multiquark systems in a nonrelativistic constituent quark model and decide whether they are bound below their lowest two-hadron threshold.

The Hamiltonian is the AL1 quark potential model (Semay and Silvestre-Brac): constituent quark rest masses and kinetic energies of all particles, plus a pairwise interaction made of a color Coulomb term, a constant, a Gaussian-smeared color-magnetic spin-spin term and a linear confinement term, all weighted by the SU(3) color factor of each pair. There is no spin-orbit or tensor force, so total spin and orbital angular momentum are separately conserved and the ground states are expected in the S wave.

A state lives in continuous particle coordinates and in a discrete space of coupled color-spin channels. Each system is fixed to one physical sector: an overall color singlet with given total spin, total isospin and positive spatial parity, obeying Fermi statistics for identical quarks (with the isospin of the light pair included). The two-hadron thresholds are fixed numbers of the same model.

Three systems are scored:

| Experiment | Particles | Sector | Lowest threshold |
|---|---|---|---|
| `tcc_al1_binding` | c c qbar qbar | I(J^P) = 0(1+) | D D* = 3878 MeV |
| `tbb_al1_binding` | b b qbar qbar | I(J^P) = 0(1+) | B B* = 10644 MeV |
| `pcc_al1_binding` | c c q q cbar | I(J^P) = 0(5/2-) | Dbar* Xi_cc* = 5718 MeV |

(q = u, d.) The doubly charmed tetraquark is expected to be a shallow, spatially extended state just below its threshold; the doubly bottom tetraquark a compact, deeply bound state; the triply heavy pentaquark is a five-body system whose binding, if any, is very shallow.

The quantity of interest is the energy of the best state one can construct in each sector: by the variational principle, any admissible normalizable state has an energy at or above the true ground state, so a lower energy means a better approximation of the ground state, and an energy below the threshold establishes a bound state. Energies are in MeV; coordinates in GeV^-1 (hbar = c = 1).
