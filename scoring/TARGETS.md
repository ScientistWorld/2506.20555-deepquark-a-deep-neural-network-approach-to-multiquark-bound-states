# Evaluation Targets

Run the accepted evaluation with:

```bash
bash scripts/evaluate.sh
```

The accepted `scores.json` contains two experiments: `tcc_al1_binding` and `tbb_al1_binding`.

## tcc_al1_binding

Tests whether a solver finds the shallow isoscalar vector doubly charmed tetraquark bound state under the AL1 Hamiltonian.

Primary target: `binding_mev`, lower is better. This is the scientific quantity the paper optimizes for this state: variational energy below the two-meson threshold.

Constraint target: `parameter_count`, lower is better at fixed quality, plus `mc_error_mev`, lower is better, and the reported color-channel proportions. A scientist should not win only by making the solver much larger, accepting high Monte Carlo noise, or dropping the coupled-channel description.

Ablation target: `threshold_baseline` in `scores.json`. This no-binding baseline scores `binding_mev = 0.0`; removing the variational multiquark treatment should fail to recover the negative binding energy.

How to run: `bash scripts/evaluate.sh`, which reads artifacts named `<method>_ccqq` from `checkpoints/` and writes `scoring/scores.json`.

## tbb_al1_binding

Tests whether a solver finds the deeply bound isoscalar vector doubly bottom tetraquark under the same AL1 Hamiltonian.

Primary target: `binding_mev`, lower is better. This checks that the method reproduces the compact heavy-tetraquark regime rather than only the shallow charm case.

Constraint target: `parameter_count`, lower is better at fixed quality, plus `mc_error_mev`, lower is better, and the reported color-channel proportions. Improvements should preserve comparable solver scale and stable Monte Carlo estimates.

Ablation target: `threshold_baseline` in `scores.json`. This no-binding baseline scores `binding_mev = 0.0`; removing the variational multiquark treatment should fail to recover the large negative binding energy.

How to run: `bash scripts/evaluate.sh`, which reads artifacts named `<method>_bbqq` from `checkpoints/` and writes `scoring/scores.json`.
