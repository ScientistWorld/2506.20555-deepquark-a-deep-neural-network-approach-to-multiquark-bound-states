# Hardening report

The gym now scores a **wavefunction**, not a report. A submission is `checkpoints/<method>_<sys>/wavefunction.jaxexport` (an exported log-psi function). The hidden test (`eval/test/evaluate.py`) owns the AL1 Hamiltonian, the sectors, the thresholds, the Metropolis sampler and its seed, and the sample count. It recomputes every scored number from fresh samples of that function and ignores every JSON a method writes. The earlier part of this iteration built that evaluator (job `ea5533c4`, results in `notes/hardening/results/`).

This last pass did the following:
- **Larger test sample.** Raised the hidden test's sample size 4x (4096 chains x 2048 samples per system) and the public train slice's 4x (x 512), because Tcc evaluator noise was as large as the effects to be resolved.
- **New reference states.** Replaced the reference with reproducible or genuinely bound states:
  - Tcc and Tbb: the `DQ_SEED=0` runs of `scripts/reproduce.sh`.
  - Pcc: the refined state that binds.
- **New weights.** Reweighted the experiments.
- **Check job.** Checked everything in job `4e44104c-455`; results are in `notes/hardening/final/`.

All scores below are hidden-test values with the final evaluator unless marked.

**How to read R.** R is the raw-unit weighted rank versus the final reference: R = sum over experiments of w_e x [1.0 x Δbinding + 0.1 x Δmc_error]. Improvements are taken in the better direction, so positive means better than the reference. Weights are Tcc 0.35, Tbb 0.30, Pcc 0.35. The gym's ranker additionally divides each metric by its spread over the ranked attempts. So only the signs and relative sizes of R are meaningful, not its units. A failed run gets `binding_mev = 0` and `mc_error_mev = 5` in every experiment, which is R = -51.39.

## Do better results now score higher?

**Before, with the redteam's JSON evaluator** (REDTEAM.md, ranks versus the shipped reports, old coefficients):
- The weaker variant scored -567.
- "More training" scored +0.38. Against its own seed-0 start that is only +0.12, below the 0.16 seed spread.
- Four times as many evaluation samples scored +0.17. Against seed 0 that is -0.09.
- The no-binding baseline outranked the reference by **+225**.

So a worse method could score higher, and a better one could not be told apart from noise.

**After:**

| Variant (same evaluator seed unless noted) | Tcc binding | Tbb binding | Pcc binding | R |
|---|---:|---:|---:|---:|
| **Reference** (seed-0 Tcc/Tbb, refined Pcc) | -13.782 ± 0.098 | -152.888 ± 0.025 | -0.591 ± 0.020 | 0 |
| Weaker: 20 + 20 + 10x2 iterations (redteam) | 0 (E = 4635) | 0 (E = 10879) | missing | **-51.39** |
| Shipped states, base-schedule Pcc | -13.181 | -152.600 | 0 (E = 5718.64, unbound) | **-0.678** |
| Better: paper's larger final-stage sample (1e5 instead of 2e4 samples, 10 x 50 SR iterations, warm start, own selection) | -13.878 | -152.900 | -0.591 (start kept) | **+0.037** |
| Same, both measured with evaluator seed 101 (paired) | -13.948 vs -13.667 | -152.875 vs -152.843 | same | **+0.109** |
| More training at lr 1e-4 (redteam, from seed 0) | -13.809 | -152.819 | — | -0.012 |
| Training seed 1 / seed 2 of the reference recipe | -12.956 / -14.012 | -152.836 / -152.949 | — | -0.306 / +0.100 |

The weaker run and the less-refined Pcc state score far below the reference. The Pcc state goes from unbound to -0.59 MeV, against 0.03 MeV of evaluator noise. Better wavefunctions score higher in every paired comparison.

The larger-sample variant beats the reference on Tcc in both paired comparisons, by +0.10 and +0.28 MeV against an evaluator noise of σ ≈ 0.06 per measurement. Its Tbb gain (+0.01 and +0.03) is within noise. On Pcc its own evaluation kept the starting state. Its Tcc gain is smaller than the training-seed spread (below).

## Major shortcuts (REDTEAM.md): fix and score before → after

The fix for all of them is one change: the evaluator measures a submitted wavefunction under its own Hamiltonian and never reads method JSON. Each one was rerun through the final hidden test in `notes/hardening/final/sc_*`.

| Shortcut | Before | After |
|---|---:|---|
| Visible-answer / threshold replay (JSON records only) | +225.10 | failed (missing wavefunction): R = -51.39 |
| Fabricated energies (JSON) | +887.98 | failed: -51.39 |
| Impossible values (JSON) | +1.28e9 | failed: -51.39 |
| Zero errors / zero parameters, changed Hamiltonian (+identity shift), moved threshold, forged scores (`metadata_claims`: reference wavefunction plus all these claims) | +297.15 / +96.00 / tie / tie | identical to the reference's own state (R = 0.000): every claim is ignored |
| Test-record replay (min of own records) | tie | not possible: the evaluator draws its own samples |
| Null / empty records | crash (no score) | failed, worst values (e.g. `empty`: "bad file size 0"), R = -51.39 |

Other attacks on the measurement itself, all through the final hidden test:

| Probe | Outcome |
|---|---|
| Malformed export (`corrupt`) | failed: cannot deserialize |
| Over the constant budget (`bloated`, +20000 constants) | failed: 21922 > 10000 |
| Fermionic sign dropped (`abs_sign`) | failed: outside the physical sector |
| Wrong exchange signs (`bosonic`) | failed: outside the physical sector |
| Output depends on call layout (`noisy`) | failed: layout check |
| Recognizes finite-difference stencils (`stencil`) | failed: layout check |
| Cusp ridges that would fool an analytic Laplacian (`kink`, new) | valid but much worse (Tcc unbound at +39 MeV, Tbb -107.9, Pcc unbound): the random-step finite-difference stencil integrates the cusp terms, R = -18.95 |
| Untrained single-Gaussian state in the right sector (`gaussian`) | valid, unbound everywhere: R = -51.39 |

## Shortcuts left open

- **Training compute is not bounded.** More training, more restarts or best-of-N seeds can give lower energies; seed 2 alone gives +0.10. This is a genuine variational improvement, since lower energy means a better state, not a reporting trick. The test cannot see training cost, and the paper's efficiency claim is not scored.
- **Procedurally generated weights** (random features computed inside the function from a few constants) are not counted against the 10000-constant budget. They are bounded only by the 30-minute limit per system. Parameter count is a validity gate and diagnostic, not a scored metric.
- **Wavefunctions designed to keep the sampler out of high-energy regions** (non-ergodic traps). These are only partly guarded, by 4096 independently started chains and the split-R-hat ≤ 1.2 gate. No working construction is known, and none was tried beyond these gates.
- **Hidden-test selection.** The test seed lives in `eval/test/`. A method that could run that file could pick among its own states by their exact test score, a gain of about the test noise (≈0.1 MeV Tcc). `eval/test/` is withheld from methods; the 4x sample cut that gain in half.

## Reference score before and after

| | Tcc binding | Tbb binding | Pcc binding |
|---|---:|---:|---:|
| Before: shipped reports, read by the old JSON evaluator (selected minimum of 10 noisy estimates) | -14.15 | -153.35 | not scored |
| Shipped states, measured by the new evaluator (job `ea5533c4`, 512 samples per chain) | -13.49 | -152.59 | 0 (unbound, +0.67) |
| Shipped states, final evaluator | -13.18 | -152.60 | 0 (unbound, +0.64) |
| **After: final reference, final evaluator** (`scoring/scores_test.json`) | **-13.78 ± 0.10** | **-152.89 ± 0.03** | **-0.59 ± 0.02** |
| Final reference, public train slice (`scoring/scores_train.json`) | -13.39 ± 0.41 | -152.66 ± 0.07 | -0.64 ± 0.04 |
| Paper (Table I/II; GEM) | -15 (GEM -14) | -153 (GEM -152) | -3 |

The method's own selection reports Tcc at 3863.16 MeV, while the evaluator measures the same state at 3864.22 MeV. Choosing the minimum of ten noisy estimates is about 1 MeV optimistic for Tcc. The old JSON score and, to a smaller extent, the paper's numbers carry this bias.

Reference provenance:
- **Tcc / Tbb.** `DQ_SEED=0` runs of `scripts/reproduce.sh` (released-code schedule), the median of three training seeds. The shipped Tbb state was worse than all three seeds, so a plain rerun would have gained about 0.3 MeV for free.
- **Pcc.** The earlier run's base schedule followed by eight 50-iteration blocks at lr 1e-4, branched from final checkpoint 8. The method's own evaluation selected checkpoint 27. `scripts/reproduce.sh` now encodes this refinement (ten blocks at lr 1e-4 from the last final checkpoint), because at lr 1e-3 alone the pentaquark hovers at its threshold.
- The training code did not change. Only seeding hooks, a continuation warm-up option and the exporter were added. The states were exported with the final `export.py`.

## Noise and headroom

| | Tcc | Tbb | Pcc |
|---|---:|---:|---:|
| Evaluator seeds (default / 101 / 202), same state | -13.78 / -13.67 / -13.69 (range 0.11) | -152.89 / -152.84 / -152.86 (range 0.05) | -0.591 / -0.586 / -0.561 (range 0.03) |
| Same, before the 4x sample (shipped Tcc/Tbb states, 4 seeds; refined Pcc state, 3 seeds) | range 0.54 | range 0.13 | range 0.03 |
| Training seeds 0 / 1 / 2 of the reference recipe | -13.78 / -12.96 / -14.01 (range 1.06) | -152.89 / -152.84 / -152.95 (range 0.11) | one run only |

- **Tbb, against GEM.** The reference is 0.89 MeV below the GEM energy, the paper's direction (the paper reports 1 MeV). That is well beyond the 0.11 seed range.
- **Tcc, against GEM.** The reference is 0.22 MeV above GEM. That is within the 1 MeV seed spread, so the reproduction does not resolve the paper's "below GEM" claim for Tcc.
- **Binding.** All three bindings exceed their noise by far: 13.8 MeV, 152.9 MeV and 0.59 MeV against noise of 0.1, 0.03 and 0.03 MeV.
- **Headroom.** The reference sits at no bound of any scored metric. `binding_mev` has no known lower bound, since the true ground states are unknown, and `mc_error_mev` is 0.02–0.10, above 0. Variants score above it: the larger-sample stage at +0.04 and +0.11, and seed 2 at +0.10. The added headroom experiment is the pentaquark, where the reproduced method falls short: -0.59 MeV against the paper's -3.1 ± 0.2 (SM-III). Its weight rose from 0.20 to 0.35.
- **Baselines and brute force.** The threshold baseline and an untrained Gaussian both score as no binding, clearly below the reference. There is no enumerable candidate set to brute-force. A basis-expansion solver (the paper's GEM, whose published Tcc is below this reference) is an honest alternative scored on the same footing, not a loophole.
- **Hidden-test cost.** About 45–55 s per system on an A100, roughly 3 minutes per method. The test never retrains. Every honest test run had split R-hat ≤ 1.009. On train the reference Tcc reached 1.118, under the 1.2 gate.

## Weights

| Experiment | Before | After | Reason |
|---|---:|---:|---|
| Tcc | 0.45 | 0.35 | |
| Tbb | 0.35 | 0.30 | Converged compact case with the least headroom; the reproduction matches the paper's 10491 MeV |
| Pcc | 0.20 | 0.35 | The paper's novel prediction ("outperforms existing calculations for pentaquarks"), and the experiment with the largest gap between the reproduction and the paper |

Metric coefficients are unchanged: `binding_mev` 1.0 and `mc_error_mev` 0.1. Parameter count, energy, validity and colors stay diagnostics at 0.

## Not fixed

- **The Pcc reference is not from a seeded run.** The from-scratch seeded `reproduce.sh` Pcc branch was started in the final job and cancelled after about 26 minutes in its first stage. It ran at about 4 s per iteration on that node and would not have fit its time box. Whether a fresh seeded run binds Pcc is therefore unverified. Its final stage alone hovers within about ±1 MeV of threshold.
- **Pcc is about 5x shallower than the paper** (-0.59 against -3 MeV). The paper's Table II row is kept unchanged in `scoring/reference.json` as `_deepquark_paper`. The Pcc `deepquark` row there holds the gym-scale value (the hidden-test reference). This is because the paper's minimum of its own checkpoint estimates after its own training is not on the gym's fresh-sample scale. The setup was not changed to approach the paper's number. `python validate.py` passes in plain, `--compare` and `--post-design` modes.
- **Training-seed noise on Tcc (about 1 MeV range) is larger than the hidden-test noise** (0.11). A rerun of the reference recipe with a lucky seed can gain about +0.2 MeV on Tcc. A single submission cannot separate that from a method improvement, and the larger-sample variant's Tcc gain is of that size.
- **Tcc does not reproduce "DeepQuark below GEM".**
