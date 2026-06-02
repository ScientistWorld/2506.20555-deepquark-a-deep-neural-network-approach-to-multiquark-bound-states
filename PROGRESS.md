# Progress

## What Works

- Paper analysis is complete for the core AL1 tetraquark and pentaquark claims.
- The authors' DeepQuark code release is vendored under `method/DeepQuark`.
- `scripts/method.sh` trains/evaluates a configured DeepQuark target and copies artifacts into `checkpoints/`.
- `scripts/evaluate.sh` reads artifacts through `eval/deepquark_scores.py` and writes `scoring/scores.json` without importing `method/`.
- GPU smoke job `e34f389c-e53` completed successfully and `python validate.py --compare` passed after method artifact scoring.
- Reference/scoring metadata now matches the authors' actual parameter counts for the vendored `ccqq` configuration.
- GPU job `a2bd95b6-2a9` reproduced the core Tcc AL1 binding claim with the actual DeepQuark VMC pipeline.

## Results

Core DeepQuark `ccqq` result from job `a2bd95b6-2a9`:

- energy: 3863.85 MeV
- binding: -14.15 MeV against the 3878 MeV threshold
- Monte Carlo error: 0.20 MeV
- parameter count: 1857
- color proportions: 0.61 antitriplet-triplet, 0.39 sextet-antisextet

Paper reference: 3863 MeV total energy and -15 MeV binding. This supports the central claim that DeepQuark finds a shallow bound AL1 Tcc state below threshold.

Earlier smoke-test DeepQuark `ccqq` result:

- energy: 2991.15 MeV
- binding: -886.85 MeV
- Monte Carlo error: 47.74 MeV

This is not physically plausible for the AL1 Tcc target and is treated only as an end-to-end execution check. The run used 512 samples and five total optimization iterations, so it is too small to estimate the variational energy reliably.

## Remaining

- Run `bbqq` with the same DeepQuark AL1 tetraquark protocol to test the heavier Tbb bound state.
- If Tbb succeeds, add `ccqqc` for broader pentaquark coverage.

## Issues

- The upstream code requires JAX, Flax, and NetKet. The container setup worked on the smoke job.
- The smoke score is overbound by far more than the reference because the run was deliberately tiny; it remains documented only as an execution check.

## Deviations from Paper

- Initial smoke job scaled down Monte Carlo chains, samples, and optimization iterations. The algorithm, Hamiltonian, symmetry projections, model family, and coupled-channel representation were unchanged. This smoke run supports `method_runs` only.
- The successful Tcc job remains scale-reduced relative to the longest paper runs, but preserved the DeepQuark procedure and reached the reported shallow-binding regime.
- The next Tbb job uses the same scaled sample/iteration schedule as the successful Tcc run.
