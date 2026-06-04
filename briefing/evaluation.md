# Evaluation

Evaluation reads method artifacts from `checkpoints/` and writes `scoring/scores.json`. It must not train, sample, or import code from `method/`.

For each experiment, the primary metric is `binding_mev`, the estimated energy minus the lowest threshold energy. Lower is better because more negative values indicate stronger binding. The evaluator also records the absolute energy, Monte Carlo error estimate, parameter count, and sampled color-channel proportions. These constraint metrics discourage trivial changes that alter the ansatz size or ignore the intended coupled-channel structure.

Run:

```bash
bash scripts/evaluate.sh
```

The standard method artifact format is a directory under `checkpoints/` named `<method>_<system>`, where `<system>` is one of `ccqq`, `bbqq`, or `ccqqc`. Each artifact directory contains:

- `parameters.json`
- `train.json`
- `eval.json`
- `results.json`

The method name is inferred from the artifact directory prefix. For example, `checkpoints/my_solver_ccqq` is scored as method `my_solver` on the `tcc_al1_binding` experiment.
