# Overview

DeepQuark applies neural-network variational Monte Carlo to multiquark bound states in constituent quark models. Its central claim is that a symmetry-informed neural wavefunction can solve tetraquark and pentaquark systems competitively with established few-body methods while handling coupled color-spin channels and complex confinement interactions.

This reproduction packages the authors' released implementation, scores AL1 binding energies for representative tetraquark and pentaquark targets, and exposes a reusable evaluator for future method improvements.

---

# Problem

Solve nonrelativistic few-body bound states in a constituent quark model for multiquark systems with fixed flavor, total spin, total isospin, parity, and color-singlet constraints.

The task is to estimate the ground-state energy of selected AL1 Hamiltonians and decide whether a state is bound relative to the lowest allowed two-hadron threshold. A valid solution must handle continuous particle coordinates and discrete color-spin-isospin coupled channels while respecting identical-particle exchange symmetry.

The initial gym targets are:

- isoscalar vector doubly charmed tetraquark `ccqq`, spin-parity `1+`
- isoscalar vector doubly bottom tetraquark `bbqq`, spin-parity `1+`
- isoscalar triply heavy pentaquark `ccqqc`, spin-parity `5/2-`

All energies are reported in MeV and binding energies are measured relative to the AL1 model thresholds listed in the paper.

---

# Method

The paper proposes DeepQuark, a deep-neural-network variational Monte Carlo method for multiquark bound states.

The Hamiltonian is the AL1 nonrelativistic constituent quark model: rest masses, kinetic energy in the center-of-mass frame, a color Coulomb term, a Gaussian-smeared hyperfine spin-spin term, and a linear confinement term. The method represents a color-singlet wavefunction in coupled color-spin-isospin bases instead of single-particle orbitals.

The neural wavefunction receives particle coordinates, pairwise distances, and one-hot coupled-channel encodings. A fully connected tanh network is multiplied by a Gaussian boundary factor, then projected with explicit identical-particle antisymmetrization and parity. Samples are drawn with Metropolis-Hastings moves combining Gaussian coordinate kicks and random discrete-channel flips. Parameters are optimized by minimizing the variational energy with stochastic reconfiguration.

This workspace vendors the authors' public `DeepQuark` code release under `method/DeepQuark` and adds only packaging, configurable run sizes, and evaluation wrappers.

---

# Evaluation

Evaluation reads method artifacts from `checkpoints/` and writes `scoring/scores.json`. It must not train, sample, or import code from `method/`.

For each experiment, the primary metric is `binding_mev`, the estimated energy minus the lowest threshold energy. Lower is better because more negative values indicate stronger binding. The evaluator also records the absolute energy, Monte Carlo error estimate, parameter count, and sampled color-channel proportions. These constraint metrics discourage trivial changes that alter the ansatz size or ignore the intended coupled-channel structure.

Run:

```bash
bash scripts/evaluate.sh
```

The standard method artifact format is a directory under `checkpoints/` containing:

- `parameters.json`
- `train.json`
- `eval.json`
- `results.json`

The evaluator currently supports artifacts named `deepquark_ccqq`, `deepquark_bbqq`, and `deepquark_ccqqc`.

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


---

# Reproduction Milestones

**Current: core_claim_plus**

## Stop Justification
- Completed at milestone `core_claim_plus`.
