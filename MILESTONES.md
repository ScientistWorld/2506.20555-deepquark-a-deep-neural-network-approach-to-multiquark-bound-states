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

### [2026-06-02 19:03] - core_claim_plus
- Audited the continued workspace and confirmed the implementation is the authors' actual DeepQuark NN-VMC method, not a surrogate.
- Verified `scripts/evaluate.sh` is artifact-based, independent from `method/`, and still passes `python validate.py --compare`.
- Fixed `scripts/reproduce.sh` so a fresh run regenerates both accepted Tcc and Tbb artifacts before scoring them.
- Removed generated DeepQuark run outputs from the tracked source tree and ignored future `method/DeepQuark/AL1/` outputs.

## Stop Justification
- Completed at milestone `core_claim_plus`.
