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

### [2026-06-02 17:17] - core_claim_plus
- GPU job `9acd5f17-604` completed the scaled AL1 `ccqqc` Pccbar training stages and produced ten `Train3_*.mpack` checkpoints.
- The job failed before evaluation because the wrapper script looked for the wrong fractional-spin run directory, so no Pccbar score was claimed.
- Added resumable evaluation support and fixed run directory formatting before resubmitting evaluation of the existing Pccbar checkpoints.

### [2026-06-02 17:20] - core_claim_plus
- GPU job `528a9437-9d3` resumed the existing Pccbar checkpoints and entered evaluation, confirming the directory fix worked.
- Evaluation failed at `para_0` because upstream pentaquark color samples were sliced with two labels instead of the three-label pentaquark color basis.
- Patched `multiquark/eval.py` to split color and spin samples using `dof_color`, then resubmitted a shorter evaluation over the first three final-stage checkpoints.

### [2026-06-02 17:23] - core_claim_plus
- GPU job `4111e39d-22c` passed the corrected pentaquark color split and failed next on spin-5/2 scalar spin reporting.
- Patched scalar spin formatting, fixed the RMS helper to reshape coordinates with `_nparticles`, and enabled RMS skipping for this scored resume run.
- Resubmitting the same three trained Pccbar checkpoints for energy/color scoring.

## Stop Justification
- Not stopped. A pentaquark run is being submitted to cover a secondary DeepQuark claim.
