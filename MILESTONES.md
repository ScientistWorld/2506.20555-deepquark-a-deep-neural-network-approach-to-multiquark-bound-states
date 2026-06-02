# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-06-02 15:43] - none
- Read the paper markdown and supplemental material.
- Identified the core method as DeepQuark: coupled-channel neural-network VMC with antisymmetry/parity projection and stochastic reconfiguration.
- Located and vendored the authors' public code release from GitHub into `method/DeepQuark`.
- Added configurable run sizes, method/evaluation scripts, reference metrics, and validated the baseline-only scoring pipeline.

### [2026-06-02 15:56] - method_runs
- GPU smoke job `e34f389c-e53` completed with exit code 0.
- The vendored DeepQuark multiquark VMC pipeline trained, evaluated, wrote artifacts in `checkpoints/deepquark_ccqq`, and produced `scoring/scores.json`.
- Smoke result was intentionally undertrained and overbound, so it does not support the core claim.

### [2026-06-02 16:58] - core_claim
- GPU job `a2bd95b6-2a9` completed the scaled AL1 `ccqq` Tcc run with the authors' DeepQuark VMC algorithm.
- Reproduced a bound Tcc state: 3863.85 MeV total energy, -14.15 MeV binding against the 3878 MeV threshold, compared with the paper's -15 MeV.
- Evaluation retained constraint metrics: Monte Carlo error 0.20 MeV, 1857 parameters, and color-channel proportions.

### [2026-06-02 17:25] - core_claim_plus
- GPU job `a48a1f9b-982` completed the scaled AL1 `bbqq` Tbb run with the same DeepQuark VMC protocol.
- Reproduced the heavier tetraquark bound state: 10490.65 MeV total energy, -153.35 MeV binding against the 10644 MeV threshold, compared with the paper's -153 MeV.
- Color proportions were also consistent with a compact Tbb state: 0.968 antitriplet-triplet and 0.032 sextet-antisextet.

## Stop Justification
- Not stopped. A pentaquark run is being submitted to cover a secondary DeepQuark claim.
