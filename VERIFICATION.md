# Verification Report

> Superseded (2026-11-11): the gym now scores exported wavefunctions (`checkpoints/<method>_<sys>/wavefunction.jaxexport`) with an evaluator-owned Hamiltonian and sampler, and scores three systems (Tcc, Tbb, Pcc). See `HARDENING.md`, `briefing/evaluation.md` and `scoring/`; the notes below describe the earlier JSON-report interface.

## Initial Assessment

The workspace started at `core_claim_plus`, and that milestone remains justified. The accepted score set contains the reproduced AL1 Tcc and Tbb tetraquark targets, both close to the paper's reported energies and binding values. The method implementation is the actual DeepQuark neural variational Monte Carlo code path, while the evaluator reads checkpoint artifacts without importing `method/`.

## Issues Found

- Minor: `briefing/problem.md` and `briefing/evaluation.md` listed `ccqqc` as a standard gym target even though `scores.json` accepts only `tcc_al1_binding` and `tbb_al1_binding`.
- Minor: color-channel proportions in `reference.json` had positive coefficients, which could reward arbitrary changes in diagnostic channel fractions. These values should be recorded for auditability, not optimized directly.
- Cosmetic: a temporary regenerated score file remained under `tmp/`.

## Fixes Applied

- Clarified `briefing/problem.md` and `briefing/evaluation.md` so the accepted scored systems are `ccqq` and `bbqq`; the pentaquark result is described as paper context, not part of the accepted baseline.
- Set color-channel metric coefficients to `0.0` in `scoring/reference.json` and updated `scoring/TARGETS.md` plus `scoring/CONSTRAINTS.md` to describe them as diagnostics.
- Removed the temporary score check file from `tmp/`.
- Re-ran `bash scripts/evaluate.sh`, `bash scripts/evaluate_train.sh`, `bash scripts/evaluate_test.sh`, and `python validate.py --compare`; all checks passed.

## Final Judgment

The workspace is ready to serve as a research gym. Improve-mode agents can build on the packaged DeepQuark implementation and artifacts. From-scratch agents can use the method-agnostic problem and evaluation docs, produce artifacts under the documented `checkpoints/<method>_<system>` interface, and be scored without depending on the reproduced method code.

- Milestone: core_claim_plus
- Ready for gym use: yes
- Confidence: high
- Key limitation: accepted scores cover Tcc and Tbb tetraquark binding only, not the pentaquark secondary claim.
