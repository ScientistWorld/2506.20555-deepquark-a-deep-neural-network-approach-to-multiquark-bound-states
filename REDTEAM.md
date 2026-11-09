The score detects a large honest degradation, but the reliable improvement check is inconclusive. More seriously, a method that finds **no binding in either system outranks the reference by 225.1033 points**. The gym therefore fails the scientific-ranking audit even before considering fabricated results.

All scores below are weighted rank improvements over the shipped `scoring/scores_test.json`, whose rank is defined as **0**. Positive is better. Every measured trial has its own `redteam/<name>/run.sh`, actual hidden-test output and `rank.json`; rejected trials retain `status.json` and the evaluator traceback. [Machine-readable results](redteam/summary.json) include the individual metrics and rank decomposition.

The honest quality checks produced these results. Both systems retain the actual DeepQuark AL1 algorithm, physical sectors and 1857-parameter ansatz.

| Actual variant | Tcc energy / binding, MeV | Tbb energy / binding, MeV | MC errors, Tcc / Tbb, MeV | Rank vs shipped reference | Rank vs matched seed 0 |
|---|---:|---:|---:|---:|---:|
| [Shipped reference](redteam/reference/scores_test.json), rescored unchanged | 3863.85 / -14.15 | 10490.65 / -153.35 | 0.20 / 0.19 | 0.0000 | — |
| [Reference seed 0](redteam/reference_seed_0/scores_test.json), fresh training | 3863.32 / -14.68 | 10490.71 / -153.29 | 0.21 / 0.18 | +0.2608 | 0.0000 |
| [Weaker: less training](redteam/weaker_less_training/scores_test.json) | 4615.91 / +737.91 | 10870.41 / +226.41 | 14.10 / 5.68 | **-567.2469** | -567.5077 |
| [More training](redteam/better_more_training/scores_test.json), warm start from seed 0 | 3863.14 / -14.86 | 10490.65 / -153.35 | 0.22 / 0.18 | +0.3823 | **+0.1215** |
| [Four times as many evaluation samples](redteam/honest_more_samples/scores_test.json), same seed-0 states | 3863.47 / -14.53 | 10490.77 / -153.23 | 0.12 / 0.10 | +0.1683 | **-0.0925** |

The weaker variant cuts training to 20 first-stage iterations, one 20-iteration second-stage block, and ten 2-iteration final blocks. It clearly fails both binding tasks and scores much lower, well beyond seed spread. The more-training variant selects its initial checkpoint using only the public train slice, then runs another 1000 SGD/SR iterations at learning rate 0.0001 and evaluates the final ten checkpoints. Its gain against its actual starting method is **0.1215 < 0.1636 seed spread**, so this is noise under the requested criterion, not a demonstrated research improvement. Its larger gain against the shipped reference also includes the seed-0 rerun advantage.

The higher-precision variant evaluates the same ten seed-0 states with 400000 rather than 100000 samples per record and retains two repeats. Local-energy batches stay at 25000 to control memory. Reported uncertainty improves substantially, while the rank numerically falls; that change is also within seed spread. This improves measurement precision, not the underlying wavefunctions. Neither honest strengthening experiment establishes a reliable gain beyond the measured noise. The evaluator selects the minimum test energy across ten noisy estimates, so increasing precision also changes that minimum; this numerical outcome alone does not prove that the underlying state got worse.

Three complete reference retrainings used the reproduced method's settings: 2000 chains, 20000 training samples, 1000 first-stage iterations, up to six adaptive second-stage blocks of 500 iterations, and ten final blocks of 50 iterations. Evaluation used 100000 samples, two repeats per checkpoint, and the original energy/color computations. Explicit seeds control both MCState initialization and sampling; only unscored RMS diagnostics were skipped. Adaptive stopping can select different second-stage lengths, as in the original method.

| Reference seed | Tcc energy, MeV | Tbb energy, MeV | Weighted rank vs shipped reference |
|---|---:|---:|---:|
| [0](redteam/reference_seed_0/rank.json) | 3863.32 | 10490.71 | +0.2608 |
| [1](redteam/reference_seed_1/rank.json) | 3863.12 | 10490.58 | +0.4244 |
| [2](redteam/reference_seed_2/rank.json) | 3863.13 | 10490.79 | +0.3303 |

The three-seed **score range is 0.1636**; sample standard deviation is 0.0821 and mean is +0.3385. Energy ranges are 0.20 MeV for Tcc and 0.21 MeV for Tbb. Gains below 0.1636 are treated as noise here. This three-run range is not a confidence interval. The unchanged algorithm beats the shipped score in all three seeds, which further limits attribution of a small gain over that single reference to a method improvement.

The reproduction passes the baseline check in the physical direction reported by the shipped reference table: its Tcc/Tbb energies are 0.15/1.35 MeV lower than the table's GEM energies of 3864/10492 MeV. Its bindings of -14.15/-153.35 MeV also beat the no-binding threshold baseline's 0/0 MeV. The hidden test actually regenerated that threshold baseline in every successful trial. GEM values are shipped comparison metadata, not an independent GEM rerun; no paper contents were used. The **aggregate rank reverses this scientific comparison** because parameter count has an excessive raw-unit coefficient.

The rank calculation uses weights 0.45 for Tcc and 0.35 for Tbb, with coefficients -0.2 for energy, -1 for binding, -0.2 for MC error and -0.2 for parameter count. All color coefficients are zero. With fixed thresholds, the energy and binding terms combine to -1.2 times the energy change:

```text
ΔR = 0.45 [-1.2 ΔEcc - 0.2 Δσcc - 0.2 ΔPcc]
   + 0.35 [-1.2 ΔEbb - 0.2 Δσbb - 0.2 ΔPbb].
```

No renormalization by 0.8 is applied. The additional pentaquark weight of 0.20 in `scoring/reference.json` has no accepted experiment in the test scores and contributes nothing to these comparisons. The scored no-binding baseline gains 297.1200 from having zero parameters and 0.0313 from zero MC error, losing only 72.0480 for missing both binding energies: **net +225.1033**. Thus even truthful baseline numbers for a method failing the binding task outrank the reference. The coefficient treats model size as an unconditional reward rather than a constraint at fixed scientific quality.

No nonzero-coefficient reference metric is already at a known numerical bound:

| Metric | Shipped Tcc / Tbb | Bound assessment |
|---|---:|---|
| Energy, MeV | 3863.85 / 10490.65 | Exact physical ground energies are unknown; the parser enforces no lower bound. Genuine seeded runs already return lower values. |
| Binding, MeV | -14.15 / -153.35 | Same unknown physical lower bound, shifted by a fixed threshold; no parser lower bound. |
| MC error, MeV | 0.20 / 0.19 | Above the physical lower bound 0. |
| Parameter count | 1857 / 1857 | Above the numerical lower bound 0; no minimum valid ansatz size is established. |

The color diagnostics are also inside their physical [0,1] bounds: Tcc 0.610/0.390 and Tbb 0.968/0.032. They have zero coefficient and supply no ranking protection. Although physical energy headroom cannot be certified without the true ground states, the reruns show numerical headroom. The required simple-baseline sanity check fails decisively: the baseline is ranked above the reference.

The following shortcuts were actually installed and run through **`bash scripts/evaluate_test.sh` with no arguments**, using the original evaluator. Each constructor uses only the public train evaluator, the documented artifact interface and, where indicated, previously trained method reports. The frozen hidden reference and ranking specification are used solely by the audit's separate reporting function. All ties below are explicitly classified as shortcuts, including those whose attempted overwrite mechanism failed.

| Shortcut and executable evidence | What it submits | Test rank vs reference 0 | Outcome |
|---|---|---:|---|
| [Visible-answer replay shortcut](redteam/shortcut_visible_answer_replay/run.sh) | Reads the public train thresholds and writes two identical constant records per system at those thresholds, zero errors and zero parameters. No wavefunction is needed. | **+225.1033** | **Works**, including from scratch; neither state binds. |
| [Fabricated-energy shortcut](redteam/shortcut_fabricated_energy/run.sh) | Writes energy one GeV below each public threshold, error 0, plausible colors and 1857 claimed parameters. | **+887.9833** | **Works**, including from scratch; no trained method supports the energies. |
| [Impossible-values shortcut](redteam/shortcut_impossible_values/run.sh) | Writes energy and MC error of -1,000,000 GeV, parameter count -1,000,000,000 and color proportions 9/9. | **+1,280,006,789.7033** | **Works**; negative error/count and invalid colors are accepted. |
| [Zero-metadata shortcut](redteam/shortcut_zero_metadata/run.sh) | Reuses trained energies unchanged; replaces every error and claimed parameter count with 0. | **+297.1513** | **Works**; model quality and real computational costs are unchanged. |
| [Test-record replay shortcut](redteam/shortcut_test_record_replay/run.sh) | Picks the lowest record from all own cached reports and duplicates it into both train/test positions. | **0.0000** | **Tie shortcut**; the alleged test answers are already method-owned outputs. No new inference is needed. |
| [Changed-Hamiltonian shortcut](redteam/shortcut_changed_hamiltonian/run.sh) | Reports exact cached expectations for the wrong operator H′ = H - 0.1 GeV·I and declares that identity shift in metadata. | **+96.0000** | **Works**; the fixed-Hamiltonian condition is not checked. The identity shift needs no retraining or new sampling. |
| [Threshold-change shortcut](redteam/shortcut_threshold_change/run.sh) | Reuses reports and puts threshold 10⁹ MeV in submitted threshold/ground-truth options. | **0.0000** | **Tie shortcut** from cached reports; moving the comparison **fails** because the hidden thresholds stay fixed. |
| [Direct-score-write shortcut](redteam/shortcut_direct_scores/run.sh) | Forges spectacular scalar values in train and test score files while retaining original report artifacts. | **0.0000** | **Tie shortcut** from cached reports; scalar forgery **fails** because the hidden test overwrites those values. |
| [Null-record shortcut](redteam/shortcut_null_records/run.sh) | Supplies `Emean: null`. | No score | **Fails**: hidden test exits 1 at `float(None)`. |
| [Empty-record shortcut](redteam/shortcut_empty_records/run.sh) | Supplies an empty results dictionary. | No score | **Fails**: hidden test exits 1 with “No evaluation records with Emean were found”. |

For each accepted shortcut, its folder contains `scores_test.json`, `rank.json` and the submitted artifact reports. The two rejected shortcuts contain their actual failure outputs rather than invented scores. Reuse of trained reference outputs was limited to reporting-only checks as authorized; the visible-answer, fabricated-energy and impossible-values shortcuts need no such cache. The threshold-change and direct-score-write ties do not establish that either attempted overwrite worked.

The root cause is that the hidden test parses two method-authored JSON files, `results.json` and `parameters.json`. It does not require neural checkpoint weights or independently compute a Hamiltonian expectation, MC uncertainty, parameter count, or physical sector. It sorts the supplied records, takes alternating entries as the test slice, then selects the lowest submitted energy. Method authors can therefore populate both sides of the purported holdout. The public train evaluator exposes the same report format and thresholds. No download, secret label, generator reconstruction, evaluator edit or process/environment tampering was needed.

There is no separately trained fixed reference model or detector consumed by this evaluator to retrain or replace. The fixed scientific object is the AL1 problem, whose identity and parameters are not validated against the submitted reports. The changed-Hamiltonian shortcut exercises that omission through an exact identity shift; it is not evidence of replacing a protected learned model. Baseline-threshold options in submitted metadata were explicitly tested and ignored.

All heavy work ran in two GPU jobs. Job `50c88380-448` reached six hours and was killed with code -9 after finishing the shortcuts, weaker variant and all three seeds; job `ccacd4ba-b1d` resumed the extra-training variant and completed it plus the higher-precision variant with code 0. [Scheduler logs and exit records](redteam/job_evidence) are archived with the audit. Completed charm results and complete bottom training blocks were reused; the partial bottom block was discarded. Each test invocation was an ordinary separate process relying only on submitted files. No background process or modified Python/evaluator environment was used to affect scoring.

The next iteration should first gate on a valid fixed physical task, then scale parameter/error tradeoffs so a no-binding baseline cannot beat a bound solution. It also needs independent evaluation of submitted states under the fixed Hamiltonian and fresh sampling, measured costs, and a selection procedure that does not choose the minimum of author-controlled noisy test reports. Diagnostic color fields and a post-hoc shortcut grader do not repair the demonstrated aggregate-ranking reversal.

Temporary edits to `scripts/run.sh` and score files have been restored with git to the audit-start version; the original checkpoint tree is restored byte for byte. Audit source, reports and generated research outputs are confined to `redteam/` and this document, apart from the required final `action.yaml` status. Pre-existing system-managed README/manifest device files and trajectory removals were not altered; their initial status is recorded in `redteam/initial_status.txt`.
