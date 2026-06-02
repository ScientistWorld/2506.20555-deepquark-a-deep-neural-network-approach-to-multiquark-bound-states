# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-06-02 17:29] - core_claim_plus
- Continued from reset milestone and audited the workspace.
- Confirmed the implementation is the paper's DeepQuark VMC method, not a surrogate.
- Confirmed evaluation reads method artifacts independently and current Tcc/Tbb scores are plausible against the paper.
- Continuing toward `secondary_claims` by submitting a Pccbar evaluation over all existing final-stage checkpoints.

### [2026-06-02 17:37] - core_claim_plus
- Job `e3faa567-af8` evaluated all ten existing Pccbar final-stage checkpoints successfully.
- Best Pccbar result improved to -0.78 MeV binding, but remains noisier and shallower than the paper's -3 MeV claim.
- Added a continuation script to resume DeepQuark final-stage VMC from the last checkpoint for another optimization attempt.

### [2026-06-02 18:01] - core_claim_plus
- Job `22c16544-0a2` completed checkpoint-9 Pccbar continuation and evaluation, but the new checkpoints degraded to higher energies.
- Current best Pccbar score is -0.45 MeV binding, so the result remains an honest partial secondary attempt rather than `secondary_claims`.
- Updated the continuation script to branch from a selected stable checkpoint and prepared a lower-learning-rate retry from checkpoint 8.

### [2026-06-02 18:33] - core_claim_plus
- Job `95a388a0-463` completed the checkpoint-8, learning-rate-0.0001 Pccbar branch and evaluated checkpoints 0-29.
- Best Pccbar result improved to -1.15 MeV binding at Train3_25, with color proportions close to the paper's pentaquark decomposition.
- Prepared another cautious continuation from Train3_25 at learning rate 0.00005 to try to approach the paper's -3 MeV secondary result.

### [2026-06-02 19:04] - core_claim_plus
- Job `e577a7e1-2a0` completed the Train3_25, learning-rate-0.00005 Pccbar refinement and evaluated checkpoints 0-39.
- Best Pccbar result improved to -1.47 MeV binding at Train3_27, still below the paper's -3 MeV secondary target.
- Added a focused checkpoint-evaluation hook and prepared a Train3_27, learning-rate-0.00002 retry over checkpoints 40-49.

### [2026-06-02 19:31] - core_claim_plus
- Job `ddf3d367-3f6` completed the Train3_27, learning-rate-0.00002 Pccbar retry and selected-checkpoint evaluation.
- Best selected result was Train3_43 at -1.12 MeV binding, worse than the previous -1.47 MeV estimate.
- Prepared a no-continuation, higher-sample reevaluation of Train3_25, Train3_27, and Train3_43 to stabilize the final Pccbar score.
