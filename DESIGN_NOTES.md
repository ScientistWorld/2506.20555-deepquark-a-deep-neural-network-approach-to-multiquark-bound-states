# Design Notes

> Superseded (2026-11-11): the gym now scores exported wavefunctions (`checkpoints/<method>_<sys>/wavefunction.jaxexport`) with an evaluator-owned Hamiltonian and sampler, and scores three systems (Tcc, Tbb, Pcc). See `HARDENING.md`, `briefing/evaluation.md` and `scoring/`; the notes below describe the earlier JSON-report interface.

## Final Wrap-Up Checks

- Confirmed `MILESTONES.md` has `**Current: core_claim_plus**` and expanded the Stop Justification with the accepted Tcc/Tbb results and the reason Pccbar does not advance the milestone.
- Aligned `scoring/TARGETS.md` and `scoring/EXPERIMENTS.md` to the accepted `scoring/scores.json` score set: `tcc_al1_binding` and `tbb_al1_binding`.
- Documented Primary, Constraint, and Ablation targets for both accepted experiments. The threshold baseline is the ablation target in each experiment, while `parameter_count`, `mc_error_mev`, and color-channel proportions are the constraint signals.
- Tightened `scoring/CONSTRAINTS.md` and `scoring/DIRECTION.md` so future improvements must solve the same AL1 tetraquark sectors rather than win by changing thresholds, Hamiltonians, resource scale, or paradigm.
- Added metric descriptions to `scoring/reference.json` for the accepted tetraquark experiments and the documented-but-unaccepted Pccbar reference entry.

## Runtime Concerns For The Next Iteration

- `scoring/reference.json` still includes `pcc_al1_binding` as paper context, but `scoring/scores.json` intentionally omits it because the current reproduction did not validate the secondary pentaquark claim.
- The train/test anti-shortcut scripts exist and pass static validation, but this wrap-up did not run their scoring because this iteration was file-editing only.
- `scripts/run.sh` currently preserves the last Pccbar verification configuration before calling `scripts/reproduce.sh`; accepted final scoring remains driven by `scripts/evaluate.sh` over Tcc and Tbb artifacts.

## Fixed Problems

- Removed the one-to-one documentation mismatch where `scoring/EXPERIMENTS.md` described `pcc_al1_binding` even though that experiment is absent from accepted `scores.json`.
- Replaced generic target prose with explicit benefit, constraint, and ablation target descriptions for each accepted experiment.
