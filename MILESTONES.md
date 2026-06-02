# Reproduction Milestones

**Current: method_runs**

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

## Stop Justification
- Not stopped. A larger Tcc convergence job is being submitted to pursue the core claim.
