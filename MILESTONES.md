# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-06-02 00:00] - method_runs
- Vendored the authors' DeepQuark implementation and verified the NN-VMC pipeline executes end-to-end.

### [2026-06-02 00:00] - core_claim
- Reproduced the AL1 isoscalar vector doubly charmed tetraquark result with the actual DeepQuark VMC method.
- Scored Tcc at 3863.85 MeV, or -14.15 MeV binding against the 3878 MeV threshold, close to the paper's -15 MeV result.

### [2026-06-02 00:00] - core_claim_plus
- Reproduced the AL1 isoscalar vector doubly bottom tetraquark setting as an additional target.
- Scored Tbb at 10490.65 MeV, or -153.35 MeV binding against the 10644 MeV threshold, close to the paper's -153 MeV result.
- Attempted the Pccbar pentaquark extension with the same DeepQuark code; it found a shallow below-threshold state but did not reproduce the paper's -3 MeV secondary binding strongly enough to claim `secondary_claims`.

## Stop Justification
- Completed at milestone `core_claim_plus`. The reusable environment validates and scores the reproduced Tcc and Tbb claims; Pccbar is documented as a partial optional target for future improvement.
