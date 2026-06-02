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
- GPU job `9acd5f17-604` completed the DeepQuark Pccbar training stages and wrote ten `Train3_*.mpack` checkpoints, but failed before evaluation/artifact copying because `scripts/method.sh` formatted fractional spin run directories incorrectly.
- GPU job `528a9437-9d3` resumed the Pccbar run and reached `para_0` evaluation, but exposed an upstream pentaquark evaluator slicing bug: the code kept only two color labels although the pentaquark color basis has three.
- GPU job `4111e39d-22c` passed the corrected pentaquark color split and then failed on scalar spin formatting for the one-component spin-5/2 basis.
- GPU job `a03cd353-981` completed Pccbar evaluation, but validation later rejected the Pccbar method score because -0.46 MeV binding is too far from the paper's -3 MeV target.

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

Attempted DeepQuark `ccqqc` Pccbar result from job `a03cd353-981`:

- energy: 5717.54 MeV
- binding: -0.46 MeV against the 5718 MeV threshold
- Monte Carlo error: 0.39 MeV
- parameter count: 2881
- color proportions: 0.219 antitriplet-antitriplet-antitriplet, 0.780 antitriplet-sextet-antitriplet, 0.001 sextet-antitriplet-antitriplet

Paper reference: 5715 MeV total energy and -3 MeV binding. This scaled run is below threshold but too shallow for the validator's paper-consistency check, so it is documented as an attempted secondary result and omitted from `scoring/scores.json`.

Earlier smoke-test DeepQuark `ccqq` result:

- energy: 2991.15 MeV
- binding: -886.85 MeV
- Monte Carlo error: 47.74 MeV

This is not physically plausible for the AL1 Tcc target and is treated only as an end-to-end execution check. The run used 512 samples and five total optimization iterations, so it is too small to estimate the variational energy reliably.

## Remaining

- Submitted an evaluation-only Pccbar follow-up to score all ten existing `Train3_*.mpack` checkpoints with two repeats, aiming to improve the shallow pentaquark binding estimate before deciding whether longer training is needed.
- Improve the Pccbar run toward the paper's -3 MeV binding by evaluating more checkpoints/repeats or training longer, then restore its method score in `scoring/scores.json`.
- Add bottom pentaquark `bbqqb` and other table entries to move beyond `core_claim_plus` toward `secondary_claims` or `majority`.

## Issues

- The upstream code requires JAX, Flax, and NetKet. The container setup worked on the smoke job.
- The smoke score is overbound by far more than the reference because the run was deliberately tiny; it remains documented only as an execution check.
- The first Pccbar job failed after training because the script searched for `ccqqc(S=2.5_I=0)` instead of the upstream-formatted `ccqqc(S=2.5_I=0.0)`. `scripts/method.sh` now derives the run path from float formatting and supports `DQ_SKIP_TRAIN=1` plus `DQ_RUN_ID` to evaluate an existing run.
- The second Pccbar job failed in `multiquark/eval.py` because all systems were sliced as if color had two discrete degrees. The evaluator now uses `dof_color` from `hi_op_.py`, preserving the two-color tetraquark split and enabling the three-color pentaquark basis.
- The third Pccbar job failed because spin-5/2 has a one-component spin basis and `spin_proportions` returns a scalar. The evaluator now formats scalar spin proportions correctly, the RMS helper uses `_nparticles`, and the resume job skips RMS expectations because they are not scored.
- The attempted Pccbar artifact reported 2881 parameters while the reference metadata records 2921 from the paper table. The evaluator preserves actual counts from `parameters.json`; future agents should keep architecture constraints fixed unless they intentionally change and document the pentaquark ansatz.

## Deviations from Paper

- Initial smoke job scaled down Monte Carlo chains, samples, and optimization iterations. The algorithm, Hamiltonian, symmetry projections, model family, and coupled-channel representation were unchanged. This smoke run supports `method_runs` only.
- The successful Tcc job remains scale-reduced relative to the longest paper runs, but preserved the DeepQuark procedure and reached the reported shallow-binding regime.
- The successful Tbb job used the same scaled sample/iteration schedule as the successful Tcc run.
- The Pccbar job used the trained first three final-stage checkpoints and one evaluation repeat to fit the one-hour resume window. RMS expectations were skipped because scoring uses energy, Monte Carlo error, parameter count, and color-channel proportions.
