# Verification Report

## Initial Assessment

The workspace started at `core_claim_plus`, and that milestone is justified. The reproduced `tcc_al1_binding` and `tbb_al1_binding` scores are close to the paper's reported AL1 tetraquark energies, and the implementation uses the paper's actual neural variational Monte Carlo code path rather than a surrogate.

## Issues Found

- Minor: the main evaluator was artifact-based but hardcoded the reproduced method name, making it less reusable for future solvers.
- Minor: method-facing artifact names appeared in method-agnostic evaluation documentation.
- Cosmetic: generated `.nv/ComputeCache` and `trajectory/` files were tracked even though `.gitignore` excludes them.

## Fixes Applied

- Updated `eval/deepquark_scores.py` to discover any `checkpoints/<method>_<system>` artifact for the supported systems, matching the train/test evaluator interface.
- Clarified `briefing/evaluation.md`, `scoring/TARGETS.md`, and `scoring/CONSTRAINTS.md` so the public scoring contract is generic and does not require future agents to use the reproduced method name.
- Removed generated `.nv/ComputeCache` and `trajectory/` files from git tracking.
- Re-ran `bash scripts/evaluate.sh`, `bash scripts/evaluate_train.sh`, `bash scripts/evaluate_test.sh`, and `python validate.py --compare`; all passed.

## Final Judgment

The workspace is ready to serve as a research gym. Improve-mode agents can build on the packaged method and existing checkpoints; from-scratch agents can use the method-agnostic problem/evaluation files and emit the documented artifact format for scoring. The main limitation is that the accepted benchmark covers two tetraquark targets; the attempted pentaquark secondary claim remains documented but is not part of the accepted score set.

- Milestone: core_claim_plus
- Ready for gym use: yes
- Confidence: high
- Key limitation: accepted scores cover Tcc and Tbb tetraquark binding only, not the pentaquark secondary claim.
