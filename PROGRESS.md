# Progress

## What Works

- Paper analysis is complete for the core AL1 tetraquark and pentaquark claims.
- The authors' DeepQuark code release is vendored under `method/DeepQuark`.
- `scripts/method.sh` trains/evaluates a configured DeepQuark target and copies artifacts into `checkpoints/`.
- `scripts/evaluate.sh` reads artifacts through `eval/deepquark_scores.py` and writes `scoring/scores.json` without importing `method/`.
- GPU smoke job `e34f389c-e53` completed successfully and `python validate.py --compare` passed after method artifact scoring.
- Reference/scoring metadata now matches the authors' actual parameter counts for the vendored `ccqq` configuration.

## Results

Smoke-test DeepQuark `ccqq` result:

- energy: 2991.15 MeV
- binding: -886.85 MeV
- Monte Carlo error: 47.74 MeV

This is not physically plausible for the AL1 Tcc target and is treated only as an end-to-end execution check. The run used 512 samples and five total optimization iterations, so it is too small to estimate the variational energy reliably.

## Remaining

- Run a larger `ccqq` job with paper-scale optimization samples and thousands of SR steps until the Tcc binding energy is near the reported shallow bound state.
- Current submission target: one-GPU `ccqq` run with 20k optimization samples, 100k check/eval samples, and multiple SR blocks.
- If core claim is reached, add `bbqq` and then `ccqqc` runs for broader coverage.

## Issues

- The upstream code requires JAX, Flax, and NetKet. The container setup worked on the smoke job.
- The smoke score is overbound by far more than the reference because the run was deliberately tiny; this must be fixed through larger sampling/optimization before claiming `core_claim`.

## Deviations from Paper

- Initial smoke job scaled down Monte Carlo chains, samples, and optimization iterations. The algorithm, Hamiltonian, symmetry projections, model family, and coupled-channel representation were unchanged. This smoke run supports `method_runs` only.
- The next job remains scale-reduced relative to the longest paper runs, but preserves the DeepQuark procedure and increases sampling enough to test the central shallow-binding claim.
