# Overview

DeepQuark applies neural-network variational Monte Carlo to multiquark bound states in constituent quark models. Its central claim is that a symmetry-informed neural wavefunction solves tetraquark and pentaquark systems competitively with (and variationally slightly below) established few-body methods such as the Gaussian expansion method, describing the molecular Tcc and the compact Tbb with one unbiased ansatz, and predicting a weakly bound triply heavy pentaquark.

This gym scores three AL1 systems: the doubly charmed tetraquark Tcc, the doubly bottom tetraquark Tbb and the triply heavy pentaquark Pccbar (S = 5/2). A submission is a wavefunction (an exported log-psi function); the evaluator samples it, measures its energy under its own fixed Hamiltonian and reports the binding energy below the fixed two-hadron threshold and its Monte Carlo error. The reference is the authors' released DeepQuark code: Tcc -13.8 MeV, Tbb -152.9 MeV and Pcc -0.6 MeV binding on the hidden test (paper: -15, -153, -3).

---

# Problem

Find the ground states of multiquark systems in a nonrelativistic constituent quark model and decide whether they are bound below their lowest two-hadron threshold.

The Hamiltonian is the AL1 quark potential model (Semay and Silvestre-Brac): constituent quark rest masses and kinetic energies of all particles, plus a pairwise interaction made of a color Coulomb term, a constant, a Gaussian-smeared color-magnetic spin-spin term and a linear confinement term, all weighted by the SU(3) color factor of each pair. There is no spin-orbit or tensor force, so total spin and orbital angular momentum are separately conserved and the ground states are expected in the S wave.

A state lives in continuous particle coordinates and in a discrete space of coupled color-spin channels. Each system is fixed to one physical sector: an overall color singlet with given total spin, total isospin and positive spatial parity, obeying Fermi statistics for identical quarks (with the isospin of the light pair included). The two-hadron thresholds are fixed numbers of the same model.

Three systems are scored:

| Experiment | Particles | Sector | Lowest threshold |
|---|---|---|---|
| `tcc_al1_binding` | c c qbar qbar | I(J^P) = 0(1+) | D D* = 3878 MeV |
| `tbb_al1_binding` | b b qbar qbar | I(J^P) = 0(1+) | B B* = 10644 MeV |
| `pcc_al1_binding` | c c q q cbar | I(J^P) = 0(5/2-) | Dbar* Xi_cc* = 5718 MeV |

(q = u, d.) The doubly charmed tetraquark is expected to be a shallow, spatially extended state just below its threshold; the doubly bottom tetraquark a compact, deeply bound state; the triply heavy pentaquark is a five-body system whose binding, if any, is very shallow.

The quantity of interest is the energy of the best state one can construct in each sector: by the variational principle, any admissible normalizable state has an energy at or above the true ground state, so a lower energy means a better approximation of the ground state, and an energy below the threshold establishes a bound state. Energies are in MeV; coordinates in GeV^-1 (hbar = c = 1).

---

# Method

The paper proposes DeepQuark, a deep-neural-network variational Monte Carlo method for multiquark bound states.

The Hamiltonian is the AL1 nonrelativistic constituent quark model: rest masses, kinetic energy, a color Coulomb term, a Gaussian-smeared hyperfine spin-spin term, and a linear confinement term. The method represents a color-singlet wavefunction in coupled color-spin-isospin bases instead of single-particle orbitals.

The neural wavefunction receives particle coordinates (center-of-mass frame), pairwise distances, and one-hot coupled-channel encodings. A fully connected tanh network (four hidden layers, about 1900 parameters for the tetraquarks and 2900 for the pentaquark) is multiplied by a Gaussian boundary factor, then projected with explicit identical-particle antisymmetrization and parity. Samples are drawn with Metropolis-Hastings moves combining Gaussian coordinate kicks and random discrete-channel flips. Parameters are optimized by minimizing the variational energy with stochastic reconfiguration; in the final stage ten checkpoints 50 steps apart are evaluated with a large sample and the lowest-energy one is kept.

This workspace vendors the authors' public `DeepQuark` code release under `method/DeepQuark` and adds only packaging, run-size settings, seeding hooks, a resumable final-stage continuation and an exporter (`multiquark/export.py`) that writes the selected state in the gym's submission format.

Reference schedule (`scripts/reproduce.sh`, released-code settings, `DQ_SEED=0`): 1000 SR iterations at lr 0.02 with 2e4 samples; blocks of 500 iterations with decaying lr until the local-energy variance check passes; ten final blocks of 50 iterations at lr 1e-3; selection of the lowest-energy final checkpoint by the method's own evaluation (2 x 1e5 samples each). For the pentaquark, whose final stage at lr 1e-3 hovers at the threshold, ten more final blocks at lr 1e-4 follow before the selection.

---

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

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 19m | **GPU**: 1.3h
- **Jobs**: 7 total (4 completed, 3 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 15:43] - none
- Read the paper markdown and supplemental material.
- Identified the core method as DeepQuark: coupled-channel neural-network VMC with antisymmetry/parity projection and stochastic reconfiguration.
- Located and vendored the authors' public code release from GitHub into `method/DeepQuark`.
- Added configurable run sizes, method/evaluation scripts, reference metrics, and validated the baseline-only scoring pipeline.

### [2026-06-02 15:56] - method_runs
- GPU smoke job `e34f389c-e53` completed with exit code 0.
- The vendored DeepQuark multiquark VMC pipeline trained, evaluated, wrote artifacts in `checkpoints/deepquark_ccqq`, and produced `scoring/scores.json`.
- Smoke result was intentionally undertrained and overbound, so it does not support the core claim.

### [2026-06-02 16:58] - core_claim
- GPU job `a2bd95b6-2a9` completed the scaled AL1 `ccqq` Tcc run with the authors' DeepQuark VMC algorithm.
- Reproduced a bound Tcc state: 3863.85 MeV total energy, -14.15 MeV binding against the 3878 MeV threshold, compared with the paper's -15 MeV.
- Evaluation retained constraint metrics: Monte Carlo error 0.20 MeV, 1857 parameters, and color-channel proportions.

### [2026-06-02 17:25] - core_claim_plus
- GPU job `a48a1f9b-982` completed the scaled AL1 `bbqq` Tbb run with the same DeepQuark VMC protocol.
- Reproduced the heavier tetraquark bound state: 10490.65 MeV total energy, -153.35 MeV binding against the 10644 MeV threshold, compared with the paper's -153 MeV.
- Color proportions were also consistent with a compact Tbb state: 0.968 antitriplet-triplet and 0.032 sextet-antisextet.

### [2026-06-02 17:17] - core_claim_plus
- GPU job `9acd5f17-604` completed the scaled AL1 `ccqqc` Pccbar training stages and produced ten `Train3_*.mpack` checkpoints.
- The job failed before evaluation because the wrapper script looked for the wrong fractional-spin run directory, so no Pccbar score was claimed.
- Added resumable evaluation support and fixed run directory formatting before resubmitting evaluation of the existing Pccbar checkpoints.

### [2026-06-02 17:20] - core_claim_plus
- GPU job `528a9437-9d3` resumed the existing Pccbar checkpoints and entered evaluation, confirming the directory fix worked.
- Evaluation failed at `para_0` because upstream pentaquark color samples were sliced with two labels instead of the three-label pentaquark color basis.
- Patched `multiquark/eval.py` to split color and spin samples using `dof_color`, then resubmitted a shorter evaluation over the first three final-stage checkpoints.

### [2026-06-02 17:23] - core_claim_plus
- GPU job `4111e39d-22c` passed the corrected pentaquark color split and failed next on spin-5/2 scalar spin reporting.
- Patched scalar spin formatting, fixed the RMS helper to reshape coordinates with `_nparticles`, and enabled RMS skipping for this scored resume run.
- Resubmitting the same three trained Pccbar checkpoints for energy/color scoring.

### [2026-06-02 17:25] - secondary_claims
- GPU job `a03cd353-981` completed evaluation and scoring for the trained AL1 `ccqqc` spin-5/2 Pccbar run.
- Reproduced the qualitative secondary claim that DeepQuark finds a weakly bound Pccbar state below the 5718 MeV threshold: 5717.54 MeV total energy and -0.46 MeV binding.
- The scaled run is shallower than the paper's -3 MeV result, but it uses the authors' pentaquark DeepQuark algorithm and preserves parameter/color-channel constraints.

</details>

### Iteration 2: gpt-5.5
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 11m | **GPU**: 1.2h
- **Jobs**: 13 total (10 completed, 3 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 17:29] - core_claim_plus
- Continued from reset milestone and audited the workspace.
- Confirmed the implementation is the paper's DeepQuark VMC method, not a surrogate.
- Confirmed evaluation reads method artifacts independently and current Tcc/Tbb scores are plausible against the paper.
- Continuing toward `secondary_claims` by submitting a Pccbar evaluation over all existing final-stage checkpoints.

### [2026-06-02 17:37] - core_claim_plus
- Job `e3faa567-af8` evaluated all ten existing Pccbar final-stage checkpoints successfully.
- Best Pccbar result improved to -0.78 MeV binding, but remains noisier and shallower than the paper's -3 MeV claim.
- Added a continuation script to resume DeepQuark final-stage VMC from the last checkpoint for another optimization attempt.

### [2026-06-02 18:01] - core_claim_plus
- Job `22c16544-0a2` completed checkpoint-9 Pccbar continuation and evaluation, but the new checkpoints degraded to higher energies.
- Current best Pccbar score is -0.45 MeV binding, so the result remains an honest partial secondary attempt rather than `secondary_claims`.
- Updated the continuation script to branch from a selected stable checkpoint and prepared a lower-learning-rate retry from checkpoint 8.

### [2026-06-02 18:33] - core_claim_plus
- Job `95a388a0-463` completed the checkpoint-8, learning-rate-0.0001 Pccbar branch and evaluated checkpoints 0-29.
- Best Pccbar result improved to -1.15 MeV binding at Train3_25, with color proportions close to the paper's pentaquark decomposition.
- Prepared another cautious continuation from Train3_25 at learning rate 0.00005 to try to approach the paper's -3 MeV secondary result.

### [2026-06-02 19:04] - core_claim_plus
- Job `e577a7e1-2a0` completed the Train3_25, learning-rate-0.00005 Pccbar refinement and evaluated checkpoints 0-39.
- Best Pccbar result improved to -1.47 MeV binding at Train3_27, still below the paper's -3 MeV secondary target.
- Added a focused checkpoint-evaluation hook and prepared a Train3_27, learning-rate-0.00002 retry over checkpoints 40-49.

### [2026-06-02 19:31] - core_claim_plus
- Job `ddf3d367-3f6` completed the Train3_27, learning-rate-0.00002 Pccbar retry and selected-checkpoint evaluation.
- Best selected result was Train3_43 at -1.12 MeV binding, worse than the previous -1.47 MeV estimate.
- Prepared a no-continuation, higher-sample reevaluation of Train3_25, Train3_27, and Train3_43 to stabilize the final Pccbar score.

### [2026-06-02 19:58] - core_claim_plus
- Job `e794595d-5d4` completed high-sample Pccbar verification over Train3_25, Train3_27, and Train3_43.
- Final Pccbar score is -0.83 MeV binding with 0.21 MeV Monte Carlo error, below threshold but too shallow for the paper's -3 MeV secondary claim.
- Reproduction is packaged and validated at `core_claim_plus`; higher milestones require stronger Pccbar optimization or additional table entries.

</details>

### Iteration 3: gpt-5.5
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 5m | **GPU**: 0.0h
- **Jobs**: 13 total (10 completed, 3 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 00:00] - method_runs
- Vendored the authors' DeepQuark implementation and verified the NN-VMC pipeline executes end-to-end.

### [2026-06-02 00:00] - core_claim
- Reproduced the AL1 isoscalar vector doubly charmed tetraquark result with the actual DeepQuark VMC method.
- Scored Tcc at 3863.85 MeV, or -14.15 MeV binding against the 3878 MeV threshold, close to the paper's -15 MeV result.

### [2026-06-02 00:00] - core_claim_plus
- Reproduced the AL1 isoscalar vector doubly bottom tetraquark setting as an additional target.
- Scored Tbb at 10490.65 MeV, or -153.35 MeV binding against the 10644 MeV threshold, close to the paper's -153 MeV result.
- Attempted the Pccbar pentaquark extension with the same DeepQuark code; it found a shallow below-threshold state but did not reproduce the paper's -3 MeV secondary binding strongly enough to claim `secondary_claims`.

### [2026-06-02 19:03] - core_claim_plus
- Audited the continued workspace and confirmed the implementation is the authors' actual DeepQuark NN-VMC method, not a surrogate.
- Verified `scripts/evaluate.sh` is artifact-based, independent from `method/`, and still passes `python validate.py --compare`.
- Fixed `scripts/reproduce.sh` so a fresh run regenerates both accepted Tcc and Tbb artifacts before scoring them.
- Removed generated DeepQuark run outputs from the tracked source tree and ignored future `method/DeepQuark/AL1/` outputs.

</details>

### Iteration 4: gpt-5.5 (designer_stage_1)
- **Milestone**: `core_claim_plus` | **Status**: error
- **Working time**: 15m | **GPU**: 0.0h
- **Jobs**: 13 total (10 completed, 3 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 00:00] - method_runs
- Vendored the authors' DeepQuark implementation and verified the NN-VMC pipeline executes end-to-end.

### [2026-06-02 00:00] - core_claim
- Reproduced the AL1 isoscalar vector doubly charmed tetraquark result with the actual DeepQuark VMC method.
- Scored Tcc at 3863.85 MeV, or -14.15 MeV binding against the 3878 MeV threshold, close to the paper's -15 MeV result.

### [2026-06-02 00:00] - core_claim_plus
- Reproduced the AL1 isoscalar vector doubly bottom tetraquark setting as an additional target.
- Scored Tbb at 10490.65 MeV, or -153.35 MeV binding against the 10644 MeV threshold, close to the paper's -153 MeV result.
- Attempted the Pccbar pentaquark extension with the same DeepQuark code; it found a shallow below-threshold state but did not reproduce the paper's -3 MeV secondary binding strongly enough to claim `secondary_claims`.

### [2026-06-02 19:03] - core_claim_plus
- Audited the continued workspace and confirmed the implementation is the authors' actual DeepQuark NN-VMC method, not a surrogate.
- Verified `scripts/evaluate.sh` is artifact-based, independent from `method/`, and still passes `python validate.py --compare`.
- Fixed `scripts/reproduce.sh` so a fresh run regenerates both accepted Tcc and Tbb artifacts before scoring them.
- Removed generated DeepQuark run outputs from the tracked source tree and ignored future `method/DeepQuark/AL1/` outputs.

</details>

### Iteration 4: gpt-5.5 (designer_stage_1)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 3m | **GPU**: 0.0h
- **Jobs**: 13 total (10 completed, 3 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 00:00] - method_runs
- Vendored the authors' DeepQuark implementation and verified the NN-VMC pipeline executes end-to-end.

### [2026-06-02 00:00] - core_claim
- Reproduced the AL1 isoscalar vector doubly charmed tetraquark result with the actual DeepQuark VMC method.
- Scored Tcc at 3863.85 MeV, or -14.15 MeV binding against the 3878 MeV threshold, close to the paper's -15 MeV result.

### [2026-06-02 00:00] - core_claim_plus
- Reproduced the AL1 isoscalar vector doubly bottom tetraquark setting as an additional target.
- Scored Tbb at 10490.65 MeV, or -153.35 MeV binding against the 10644 MeV threshold, close to the paper's -153 MeV result.
- Attempted the Pccbar pentaquark extension with the same DeepQuark code; it found a shallow below-threshold state but did not reproduce the paper's -3 MeV secondary binding strongly enough to claim `secondary_claims`.

### [2026-06-02 19:03] - core_claim_plus
- Audited the continued workspace and confirmed the implementation is the authors' actual DeepQuark NN-VMC method, not a surrogate.
- Verified `scripts/evaluate.sh` is artifact-based, independent from `method/`, and still passes `python validate.py --compare`.
- Fixed `scripts/reproduce.sh` so a fresh run regenerates both accepted Tcc and Tbb artifacts before scoring them.
- Removed generated DeepQuark run outputs from the tracked source tree and ignored future `method/DeepQuark/AL1/` outputs.

</details>

### Iteration 5: gpt-5.5 (designer_stage_2)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 15m | **GPU**: 0.0h
- **Jobs**: 13 total (10 completed, 3 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 00:00] - method_runs
- Vendored the authors' DeepQuark implementation and verified the NN-VMC pipeline executes end-to-end.

### [2026-06-02 00:00] - core_claim
- Reproduced the AL1 isoscalar vector doubly charmed tetraquark result with the actual DeepQuark VMC method.
- Scored Tcc at 3863.85 MeV, or -14.15 MeV binding against the 3878 MeV threshold, close to the paper's -15 MeV result.

### [2026-06-02 00:00] - core_claim_plus
- Reproduced the AL1 isoscalar vector doubly bottom tetraquark setting as an additional target.
- Scored Tbb at 10490.65 MeV, or -153.35 MeV binding against the 10644 MeV threshold, close to the paper's -153 MeV result.
- Attempted the Pccbar pentaquark extension with the same DeepQuark code; it found a shallow below-threshold state but did not reproduce the paper's -3 MeV secondary binding strongly enough to claim `secondary_claims`.

### [2026-06-02 19:03] - core_claim_plus
- Audited the continued workspace and confirmed the implementation is the authors' actual DeepQuark NN-VMC method, not a surrogate.
- Verified `scripts/evaluate.sh` is artifact-based, independent from `method/`, and still passes `python validate.py --compare`.
- Fixed `scripts/reproduce.sh` so a fresh run regenerates both accepted Tcc and Tbb artifacts before scoring them.
- Removed generated DeepQuark run outputs from the tracked source tree and ignored future `method/DeepQuark/AL1/` outputs.

</details>

### Iteration 6: gpt-5.5 (designer_stage_3)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 15m | **GPU**: 0.0h
- **Jobs**: 13 total (10 completed, 3 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 00:00] - method_runs
- Vendored the authors' DeepQuark implementation and verified the NN-VMC pipeline executes end-to-end.

### [2026-06-02 00:00] - core_claim
- Reproduced the AL1 isoscalar vector doubly charmed tetraquark result with the actual DeepQuark VMC method.
- Scored Tcc at 3863.85 MeV, or -14.15 MeV binding against the 3878 MeV threshold, close to the paper's -15 MeV result.

### [2026-06-02 00:00] - core_claim_plus
- Reproduced the AL1 isoscalar vector doubly bottom tetraquark setting as an additional target.
- Scored Tbb at 10490.65 MeV, or -153.35 MeV binding against the 10644 MeV threshold, close to the paper's -153 MeV result.
- Attempted the Pccbar pentaquark extension with the same DeepQuark code; it found a shallow below-threshold state but did not reproduce the paper's -3 MeV secondary binding strongly enough to claim `secondary_claims`.

### [2026-06-02 19:03] - core_claim_plus
- Audited the continued workspace and confirmed the implementation is the authors' actual DeepQuark NN-VMC method, not a surrogate.
- Verified `scripts/evaluate.sh` is artifact-based, independent from `method/`, and still passes `python validate.py --compare`.
- Fixed `scripts/reproduce.sh` so a fresh run regenerates both accepted Tcc and Tbb artifacts before scoring them.
- Removed generated DeepQuark run outputs from the tracked source tree and ignored future `method/DeepQuark/AL1/` outputs.

</details>

### Iteration 7: gpt-5.5 (packager)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 9m | **GPU**: 0.0h
- **Jobs**: 13 total (10 completed, 3 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 00:00] - method_runs
- Vendored the authors' DeepQuark implementation and verified the NN-VMC pipeline executes end-to-end.

### [2026-06-02 00:00] - core_claim
- Reproduced the AL1 isoscalar vector doubly charmed tetraquark result with the actual DeepQuark VMC method.
- Scored Tcc at 3863.85 MeV, or -14.15 MeV binding against the 3878 MeV threshold, close to the paper's -15 MeV result.

### [2026-06-02 00:00] - core_claim_plus
- Reproduced the AL1 isoscalar vector doubly bottom tetraquark setting as an additional target.
- Scored Tbb at 10490.65 MeV, or -153.35 MeV binding against the 10644 MeV threshold, close to the paper's -153 MeV result.
- Attempted the Pccbar pentaquark extension with the same DeepQuark code; it found a shallow below-threshold state but did not reproduce the paper's -3 MeV secondary binding strongly enough to claim `secondary_claims`.

### [2026-06-02 19:03] - core_claim_plus
- Audited the continued workspace and confirmed the implementation is the authors' actual DeepQuark NN-VMC method, not a surrogate.
- Verified `scripts/evaluate.sh` is artifact-based, independent from `method/`, and still passes `python validate.py --compare`.
- Fixed `scripts/reproduce.sh` so a fresh run regenerates both accepted Tcc and Tbb artifacts before scoring them.
- Removed generated DeepQuark run outputs from the tracked source tree and ignored future `method/DeepQuark/AL1/` outputs.

</details>

### Iteration 8: gpt-5.5 (verifier)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 6m | **GPU**: 0.0h
- **Jobs**: 13 total (10 completed, 3 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 00:00] - method_runs
- Vendored the authors' DeepQuark implementation and verified the NN-VMC pipeline executes end-to-end.

### [2026-06-02 00:00] - core_claim
- Reproduced the AL1 isoscalar vector doubly charmed tetraquark result with the actual DeepQuark VMC method.
- Scored Tcc at 3863.85 MeV, or -14.15 MeV binding against the 3878 MeV threshold, close to the paper's -15 MeV result.

### [2026-06-02 00:00] - core_claim_plus
- Reproduced the AL1 isoscalar vector doubly bottom tetraquark setting as an additional target.
- Scored Tbb at 10490.65 MeV, or -153.35 MeV binding against the 10644 MeV threshold, close to the paper's -153 MeV result.
- Attempted the Pccbar pentaquark extension with the same DeepQuark code; it found a shallow below-threshold state but did not reproduce the paper's -3 MeV secondary binding strongly enough to claim `secondary_claims`.

### [2026-06-02 19:03] - core_claim_plus
- Audited the continued workspace and confirmed the implementation is the authors' actual DeepQuark NN-VMC method, not a surrogate.
- Verified `scripts/evaluate.sh` is artifact-based, independent from `method/`, and still passes `python validate.py --compare`.
- Fixed `scripts/reproduce.sh` so a fresh run regenerates both accepted Tcc and Tbb artifacts before scoring them.
- Removed generated DeepQuark run outputs from the tracked source tree and ignored future `method/DeepQuark/AL1/` outputs.

### [2026-06-04 07:02] - core_claim_plus
- Final verification confirmed the claimed milestone remains justified.
- Generalized the main evaluator to score any `checkpoints/<method>_<system>` artifact for the accepted systems.
- Re-ran the accepted, train-slice, and test-slice evaluators; validation passed.

</details>

### Iteration 9: gpt-5.5 (verifier)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 8m | **GPU**: 0.0h
- **Jobs**: 13 total (10 completed, 3 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 00:00] - method_runs
- Vendored the authors' DeepQuark implementation and verified the NN-VMC pipeline executes end-to-end.

### [2026-06-02 00:00] - core_claim
- Reproduced the AL1 isoscalar vector doubly charmed tetraquark result with the actual DeepQuark VMC method.
- Scored Tcc at 3863.85 MeV, or -14.15 MeV binding against the 3878 MeV threshold, close to the paper's -15 MeV result.

### [2026-06-02 00:00] - core_claim_plus
- Reproduced the AL1 isoscalar vector doubly bottom tetraquark setting as an additional target.
- Scored Tbb at 10490.65 MeV, or -153.35 MeV binding against the 10644 MeV threshold, close to the paper's -153 MeV result.
- Attempted the Pccbar pentaquark extension with the same DeepQuark code; it found a shallow below-threshold state but did not reproduce the paper's -3 MeV secondary binding strongly enough to claim `secondary_claims`.

### [2026-06-02 19:03] - core_claim_plus
- Audited the continued workspace and confirmed the implementation is the authors' actual DeepQuark NN-VMC method, not a surrogate.
- Verified `scripts/evaluate.sh` is artifact-based, independent from `method/`, and still passes `python validate.py --compare`.
- Fixed `scripts/reproduce.sh` so a fresh run regenerates both accepted Tcc and Tbb artifacts before scoring them.
- Removed generated DeepQuark run outputs from the tracked source tree and ignored future `method/DeepQuark/AL1/` outputs.

### [2026-06-04 07:02] - core_claim_plus
- Final verification confirmed the claimed milestone remains justified.
- Generalized the main evaluator to score any `checkpoints/<method>_<system>` artifact for the accepted systems.
- Re-ran the accepted, train-slice, and test-slice evaluators; validation passed.

### [2026-06-04 07:07] - core_claim_plus
- Re-audited the workspace after handoff and confirmed the accepted score set still regenerates exactly from checkpoint artifacts.
- Verified `python validate.py --compare` passes and that `eval/` remains independent of `method/`.
- Kept Pccbar as a documented partial secondary attempt rather than an accepted target because the reproduced binding remains too shallow versus the paper.

### [2026-06-04 07:11] - core_claim_plus
- Final verification kept the accepted gym scope to the reproduced Tcc and Tbb experiments.
- Adjusted reference metric coefficients so color-channel proportions are recorded diagnostics rather than direct optimization objectives.
- Re-ran accepted, train-slice, and test-slice evaluators; validation passed.

</details>

### Iteration 10: gpt-6.1-sol (redteam)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 18m | **GPU**: 6.3h
- **Jobs**: 13 total (10 completed, 3 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 00:00] - method_runs
- Vendored the authors' DeepQuark implementation and verified the NN-VMC pipeline executes end-to-end.

### [2026-06-02 00:00] - core_claim
- Reproduced the AL1 isoscalar vector doubly charmed tetraquark result with the actual DeepQuark VMC method.
- Scored Tcc at 3863.85 MeV, or -14.15 MeV binding against the 3878 MeV threshold, close to the paper's -15 MeV result.

### [2026-06-02 00:00] - core_claim_plus
- Reproduced the AL1 isoscalar vector doubly bottom tetraquark setting as an additional target.
- Scored Tbb at 10490.65 MeV, or -153.35 MeV binding against the 10644 MeV threshold, close to the paper's -153 MeV result.
- Attempted the Pccbar pentaquark extension with the same DeepQuark code; it found a shallow below-threshold state but did not reproduce the paper's -3 MeV secondary binding strongly enough to claim `secondary_claims`.

### [2026-06-02 19:03] - core_claim_plus
- Audited the continued workspace and confirmed the implementation is the authors' actual DeepQuark NN-VMC method, not a surrogate.
- Verified `scripts/evaluate.sh` is artifact-based, independent from `method/`, and still passes `python validate.py --compare`.
- Fixed `scripts/reproduce.sh` so a fresh run regenerates both accepted Tcc and Tbb artifacts before scoring them.
- Removed generated DeepQuark run outputs from the tracked source tree and ignored future `method/DeepQuark/AL1/` outputs.

### [2026-06-04 07:02] - core_claim_plus
- Final verification confirmed the claimed milestone remains justified.
- Generalized the main evaluator to score any `checkpoints/<method>_<system>` artifact for the accepted systems.
- Re-ran the accepted, train-slice, and test-slice evaluators; validation passed.

### [2026-06-04 07:07] - core_claim_plus
- Re-audited the workspace after handoff and confirmed the accepted score set still regenerates exactly from checkpoint artifacts.
- Verified `python validate.py --compare` passes and that `eval/` remains independent of `method/`.
- Kept Pccbar as a documented partial secondary attempt rather than an accepted target because the reproduced binding remains too shallow versus the paper.

### [2026-06-04 07:11] - core_claim_plus
- Final verification kept the accepted gym scope to the reproduced Tcc and Tbb experiments.
- Adjusted reference metric coefficients so color-channel proportions are recorded diagnostics rather than direct optimization objectives.
- Re-ran accepted, train-slice, and test-slice evaluators; validation passed.

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `core_claim_plus` | **Status**: error
- **Working time**: 30m | **GPU**: 0.7h
- **Jobs**: 17 total (11 completed, 3 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 00:00] - method_runs
- Vendored the authors' DeepQuark implementation and verified the NN-VMC pipeline executes end-to-end.

### [2026-06-02 00:00] - core_claim
- Reproduced the AL1 isoscalar vector doubly charmed tetraquark result with the actual DeepQuark VMC method.
- Scored Tcc at 3863.85 MeV, or -14.15 MeV binding against the 3878 MeV threshold, close to the paper's -15 MeV result.

### [2026-06-02 00:00] - core_claim_plus
- Reproduced the AL1 isoscalar vector doubly bottom tetraquark setting as an additional target.
- Scored Tbb at 10490.65 MeV, or -153.35 MeV binding against the 10644 MeV threshold, close to the paper's -153 MeV result.
- Attempted the Pccbar pentaquark extension with the same DeepQuark code; it found a shallow below-threshold state but did not reproduce the paper's -3 MeV secondary binding strongly enough to claim `secondary_claims`.

### [2026-06-02 19:03] - core_claim_plus
- Audited the continued workspace and confirmed the implementation is the authors' actual DeepQuark NN-VMC method, not a surrogate.
- Verified `scripts/evaluate.sh` is artifact-based, independent from `method/`, and still passes `python validate.py --compare`.
- Fixed `scripts/reproduce.sh` so a fresh run regenerates both accepted Tcc and Tbb artifacts before scoring them.
- Removed generated DeepQuark run outputs from the tracked source tree and ignored future `method/DeepQuark/AL1/` outputs.

### [2026-06-04 07:02] - core_claim_plus
- Final verification confirmed the claimed milestone remains justified.
- Generalized the main evaluator to score any `checkpoints/<method>_<system>` artifact for the accepted systems.
- Re-ran the accepted, train-slice, and test-slice evaluators; validation passed.

### [2026-06-04 07:07] - core_claim_plus
- Re-audited the workspace after handoff and confirmed the accepted score set still regenerates exactly from checkpoint artifacts.
- Verified `python validate.py --compare` passes and that `eval/` remains independent of `method/`.
- Kept Pccbar as a documented partial secondary attempt rather than an accepted target because the reproduced binding remains too shallow versus the paper.

### [2026-06-04 07:11] - core_claim_plus
- Final verification kept the accepted gym scope to the reproduced Tcc and Tbb experiments.
- Adjusted reference metric coefficients so color-channel proportions are recorded diagnostics rather than direct optimization objectives.
- Re-ran accepted, train-slice, and test-slice evaluators; validation passed.

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 23m | **GPU**: 0.0h
- **Jobs**: 18 total (11 completed, 3 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 00:00] - method_runs
- Vendored the authors' DeepQuark implementation and verified the NN-VMC pipeline executes end-to-end.

### [2026-06-02 00:00] - core_claim
- Reproduced the AL1 isoscalar vector doubly charmed tetraquark result with the actual DeepQuark VMC method.
- Scored Tcc at 3863.85 MeV, or -14.15 MeV binding against the 3878 MeV threshold, close to the paper's -15 MeV result.

### [2026-06-02 00:00] - core_claim_plus
- Reproduced the AL1 isoscalar vector doubly bottom tetraquark setting as an additional target.
- Scored Tbb at 10490.65 MeV, or -153.35 MeV binding against the 10644 MeV threshold, close to the paper's -153 MeV result.
- Attempted the Pccbar pentaquark extension with the same DeepQuark code; it found a shallow below-threshold state but did not reproduce the paper's -3 MeV secondary binding strongly enough to claim `secondary_claims`.

### [2026-06-02 19:03] - core_claim_plus
- Audited the continued workspace and confirmed the implementation is the authors' actual DeepQuark NN-VMC method, not a surrogate.
- Verified `scripts/evaluate.sh` is artifact-based, independent from `method/`, and still passes `python validate.py --compare`.
- Fixed `scripts/reproduce.sh` so a fresh run regenerates both accepted Tcc and Tbb artifacts before scoring them.
- Removed generated DeepQuark run outputs from the tracked source tree and ignored future `method/DeepQuark/AL1/` outputs.

### [2026-06-04 07:02] - core_claim_plus
- Final verification confirmed the claimed milestone remains justified.
- Generalized the main evaluator to score any `checkpoints/<method>_<system>` artifact for the accepted systems.
- Re-ran the accepted, train-slice, and test-slice evaluators; validation passed.

### [2026-06-04 07:07] - core_claim_plus
- Re-audited the workspace after handoff and confirmed the accepted score set still regenerates exactly from checkpoint artifacts.
- Verified `python validate.py --compare` passes and that `eval/` remains independent of `method/`.
- Kept Pccbar as a documented partial secondary attempt rather than an accepted target because the reproduced binding remains too shallow versus the paper.

### [2026-06-04 07:11] - core_claim_plus
- Final verification kept the accepted gym scope to the reproduced Tcc and Tbb experiments.
- Adjusted reference metric coefficients so color-channel proportions are recorded diagnostics rather than direct optimization objectives.
- Re-ran accepted, train-slice, and test-slice evaluators; validation passed.

### [2026-11-11 00:00] - core_claim_plus
- Hardening: the gym now scores exported wavefunctions with an evaluator-owned AL1 Hamiltonian, sampler and gates; every redteam shortcut fails or ties its own reference state (HARDENING.md).
- Hidden-test sample raised 4x; reference is the seed-0 Tcc/Tbb runs and the lr-1e-4-refined Pcc run: Tcc -13.78, Tbb -152.89, Pcc -0.59 MeV binding (paper -15, -153, -3).
- Pcc stays short of the paper's -3 MeV binding, so `secondary_claims` is not claimed.

</details>


---

# Reproduction Milestones

**Current: core_claim_plus**

## Stop Justification
- Completed at milestone `core_claim_plus`.
