# Evaluation Targets

Score every method under `checkpoints/` with

```bash
bash scripts/evaluate_train.sh   # public slice  -> scoring/scores_train.json
bash scripts/evaluate_test.sh    # hidden slice  -> scoring/scores_test.json (held out)
bash scripts/evaluate.sh         # public slice  -> scoring/scores.json, then validate.py
```

Each experiment reads `checkpoints/<method>_<suffix>/wavefunction.jaxexport` (see `briefing/evaluation.md`); nothing else a method writes is read.

| Experiment | Suffix | Weight | Reference (hidden test) | Paper |
|---|---|---|---|---|
| `tcc_al1_binding` | `ccqq` | 0.35 | -13.78 MeV (error 0.10) | -15 (GEM -14) |
| `tbb_al1_binding` | `bbqq` | 0.30 | -152.89 MeV (error 0.03) | -153 (GEM -152) |
| `pcc_al1_binding` | `ccqqc` | 0.35 | -0.59 MeV (error 0.02) | -3 |

## Per experiment

Primary target: `binding_mev`, lower is better. The variational energy measured by the evaluator minus the fixed threshold; 0 if the state is not bound.

Secondary target: `mc_error_mev` (coefficient 0.1), lower is better. Precision of that energy at the evaluator's fixed sample count.

Constraints (validity gates; a violation scores the run as failed, i.e. `binding_mev = 0`, `mc_error_mev = 5`): the fixed AL1 Hamiltonian, thresholds, sector (exchange symmetry and parity) and sampler are the evaluator's; the submitted function must be deterministic, pure, layout-independent and within 10000 stored float64-equivalent constants; the sampler must converge (split R-hat <= 1.2); the energy must not be implausibly far below threshold.

Diagnostics (not ranked): `energy_mev`, `parameter_count`, `valid`, color-channel proportions.

Baseline: `threshold_baseline` (no binding: `binding_mev = 0`, `mc_error_mev = 5`) is written by the evaluator for every experiment. An untrained single-Gaussian state in the right sector is unbound in all three systems (Tcc 4840, Tbb 11360, Pcc 7834 MeV), and a shortened training run is unbound in Tcc and Tbb.
