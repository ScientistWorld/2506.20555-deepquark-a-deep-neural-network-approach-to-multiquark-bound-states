# Evaluation

A method submits, for each system, a wavefunction: a serialized function that the evaluator samples and measures itself. Nothing else a method writes is read; reported energies, errors, parameter counts or thresholds in any JSON file are ignored.

## Submission interface

For method `<method>` and system suffix `ccqq`, `bbqq` or `ccqqc` (Tcc, Tbb, Pcc), write

    checkpoints/<method>_<suffix>/wavefunction.jaxexport

containing `jax.export.export(jax.jit(f), platforms=("cpu", "cuda"))(spec).serialize()` of a function

    f(x: float64[batch, dim]) -> complex128[batch] or float64[batch]    # log psi

* Each row of `x` is the particle coordinates in GeV^-1, particle-major (`x0, y0, z0, x1, ...`), followed by the one-hot code of one coupled color-spin channel. Particle order follows the suffix: `ccqq`/`bbqq` = heavy quark, heavy quark, light antiquark, light antiquark (dim 3*4 + 5 = 17); `ccqqc` = c, c, q, q, cbar (dim 3*5 + 4 = 19).
* The channel codes, the AL1 parameters and the color/spin matrix elements (including the phase convention of every channel) are defined in the public `eval/train/evaluate.py` (`_tetra_tables`, `_penta_tables`, `AL1`, `QUARK_MASS`). The evaluator builds the Hamiltonian from them.
* `batch` may be symbolic or a fixed size that divides 4096. The function must be deterministic and pure (no callbacks, random-number ops or other side effects), single-device, and exported for the CUDA platform. Its stored constants may total at most 10000 float64-equivalents (8 bytes each).
* The method name is the directory prefix; every method found under `checkpoints/` is scored on all three systems, and a missing system scores as failed.

## Measurement

For each submission the evaluator runs its own Metropolis sampler of |psi|^2 over coordinates and channels (4096 chains, fixed seed, adaptive step size, burn-in, thinning) and computes local energies with its own AL1 Hamiltonian: a finite-difference Laplacian with randomly oriented, randomly sized steps for the kinetic term, and the full channel-coupled potential. The function is only ever called on batches that mix rows of many different samples.

The run is scored as **failed** if the file is missing or malformed, violates the interface above, is over the parameter budget, returns non-finite local energies, is not deterministic, returns different values for the same configuration in different calls or call layouts, violates the required exchange symmetry or parity (checked on 65536 samples), does not converge in the sampler (split R-hat > 1.2), reports an energy far below threshold (more than 100, 400, 100 MeV for Tcc, Tbb, Pcc), or exceeds 30 minutes.

## Metrics

| Metric | Scored | Better | Meaning |
|---|---|---|---|
| `binding_mev` | yes (coefficient 1.0) | lower | measured energy minus the fixed threshold; 0 if not bound |
| `mc_error_mev` | yes (coefficient 0.1) | lower | Monte Carlo standard error of that energy from the evaluator's chains, capped at 5 |
| `energy_mev` | diagnostic | lower | measured total energy |
| `parameter_count` | diagnostic | — | stored constants in float64 equivalents (gated at 10000) |
| `valid` | diagnostic | — | 1 if every check passed |
| `color_*` | diagnostic | — | sampled color-channel proportions |

An unbound, failed or missing submission gets the worst value of both scored metrics (`binding_mev = 0`, `mc_error_mev = 5`). The fixed `threshold_baseline` row (a state at the threshold, no binding) is written alongside every method.

## Running

```bash
bash scripts/evaluate_train.sh   # public slice -> scoring/scores_train.json
bash scripts/evaluate.sh         # same protocol -> scoring/scores.json, then validate.py
```

The public train slice uses a public sampler seed and 4096 x 512 samples per system (about a minute per system on one GPU). The hidden test measures the same quantities with its own seed and a larger sample, so an honest wavefunction scores alike on both up to Monte Carlo noise (on train, roughly 0.1-0.4 MeV for Tcc and below 0.1 MeV for Tbb and Pcc). Experiment weights: Tcc 0.35, Tbb 0.30, Pcc 0.35.
