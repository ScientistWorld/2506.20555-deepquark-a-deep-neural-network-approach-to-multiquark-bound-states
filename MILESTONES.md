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

### [2026-06-04 07:02] - core_claim_plus
- Final verification confirmed the claimed milestone remains justified.
- Generalized the main evaluator to score any `checkpoints/<method>_<system>` artifact for the accepted systems.
- Re-ran the accepted, train-slice, and test-slice evaluators; validation passed.

### [2026-06-04 07:07] - core_claim_plus
- Re-audited the workspace after handoff and confirmed the accepted score set still regenerates exactly from checkpoint artifacts.
- Verified `python validate.py --compare` passes and that `eval/` remains independent of `method/`.
- Kept Pccbar as a documented partial secondary attempt rather than an accepted target because the reproduced binding remains too shallow versus the paper.

### [2026-06-04 07:11] - core_claim_plus
- Final verification kept the accepted gym scope to the reproduced Tcc and Tbb experiments.
- Adjusted reference metric coefficients so color-channel proportions are recorded diagnostics rather than direct optimization objectives.
- Re-ran accepted, train-slice, and test-slice evaluators; validation passed.

### [2026-11-11 00:00] - core_claim_plus
- Hardening: the gym now scores exported wavefunctions with an evaluator-owned AL1 Hamiltonian, sampler and gates; every redteam shortcut fails or ties its own reference state (HARDENING.md).
- Hidden-test sample raised 4x; reference is the seed-0 Tcc/Tbb runs and the lr-1e-4-refined Pcc run: Tcc -13.78, Tbb -152.89, Pcc -0.59 MeV binding (paper -15, -153, -3).
- Pcc stays short of the paper's -3 MeV binding, so `secondary_claims` is not claimed.

## Stop Justification
- Completed at milestone `core_claim_plus`.
