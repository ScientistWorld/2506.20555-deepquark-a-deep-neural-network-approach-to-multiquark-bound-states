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
