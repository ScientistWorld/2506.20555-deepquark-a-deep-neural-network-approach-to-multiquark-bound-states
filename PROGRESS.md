# Progress

## What Works

- Paper analysis is complete for the core AL1 tetraquark and pentaquark claims.
- The authors' DeepQuark code release is vendored under `method/DeepQuark`.
- `scripts/method.sh` trains/evaluates a configured DeepQuark target and copies artifacts into `checkpoints/`.
- `scripts/evaluate.sh` reads artifacts through `eval/deepquark_scores.py` and writes `scoring/scores.json` without importing `method/`.
- GPU smoke job `e34f389c-e53` completed successfully and `python validate.py --compare` passed after method artifact scoring.
- Reference/scoring metadata now matches the authors' actual parameter counts for the vendored `ccqq` configuration.
- GPU job `a2bd95b6-2a9` reproduced the core Tcc AL1 binding claim with the actual DeepQuark VMC pipeline.
- GPU job `a48a1f9b-982` reproduced the Tbb AL1 tetraquark result with the same pipeline, extending the reproduction beyond the minimum.

## Results

Core DeepQuark `ccqq` result from job `a2bd95b6-2a9`:

- energy: 3863.85 MeV
- binding: -14.15 MeV against the 3878 MeV threshold
- Monte Carlo error: 0.20 MeV
- parameter count: 1857
- color proportions: 0.61 antitriplet-triplet, 0.39 sextet-antisextet

Paper reference: 3863 MeV total energy and -15 MeV binding. This supports the central claim that DeepQuark finds a shallow bound AL1 Tcc state below threshold.

Additional DeepQuark `bbqq` result from job `a48a1f9b-982`:

- energy: 10490.65 MeV
- binding: -153.35 MeV against the 10644 MeV threshold
- Monte Carlo error: 0.19 MeV
- parameter count: 1857
- color proportions: 0.968 antitriplet-triplet, 0.032 sextet-antisextet

Paper reference: 10491 MeV total energy and -153 MeV binding. This supports `core_claim_plus` by reproducing the compact Tbb tetraquark setting.

Earlier smoke-test DeepQuark `ccqq` result:

- energy: 2991.15 MeV
- binding: -886.85 MeV
- Monte Carlo error: 47.74 MeV

This is not physically plausible for the AL1 Tcc target and is treated only as an end-to-end execution check. The run used 512 samples and five total optimization iterations, so it is too small to estimate the variational energy reliably.

## Remaining

- Run `ccqqc` with the DeepQuark AL1 pentaquark protocol to test the reported weakly bound Pccbar state.
- If Pccbar succeeds, consider the bottom pentaquark `bbqqb` for additional secondary coverage.

## Issues

- The upstream code requires JAX, Flax, and NetKet. The container setup worked on the smoke job.
- The smoke score is overbound by far more than the reference because the run was deliberately tiny; it remains documented only as an execution check.

## Deviations from Paper

- Initial smoke job scaled down Monte Carlo chains, samples, and optimization iterations. The algorithm, Hamiltonian, symmetry projections, model family, and coupled-channel representation were unchanged. This smoke run supports `method_runs` only.
- The successful Tcc job remains scale-reduced relative to the longest paper runs, but preserved the DeepQuark procedure and reached the reported shallow-binding regime.
- The successful Tbb job used the same scaled sample/iteration schedule as the successful Tcc run.
- The next Pccbar job keeps that schedule but switches to the authors' pentaquark architecture and spin-5/2 sector.
