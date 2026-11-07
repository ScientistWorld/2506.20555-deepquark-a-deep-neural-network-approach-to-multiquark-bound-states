# Gym scoring audit

`python3 redteam/run_all.py` runs the full bundle on a compute node. Each named
trial also has an independent `run.sh`, saves the real hidden-test JSON and a
weighted rank delta against the shipped test reference, and restores submission
artifacts afterwards. A failed test is recorded as a failure, never as a score.

Shortcut constructors use only the documented artifact interface, the public
train evaluator and already trained method outputs. They do not read the hidden
evaluator, hidden scores, paper, or ranking specification. The separate reporting
function uses the frozen specification solely to calculate audit ranks. The
cached reference outputs are reused for checks that only alter reporting,
as authorized in the audit request. Those shortcuts are conditional on having
run a solver before; the visible-answer and fabricated-answer shortcuts also
work from scratch and require no trained model.

Quality variants copy the existing method into ignored work directories and
add explicit seeds to NetKet MCState initialization and sampling. Reference
reruns use the original 2000 chains, 20000 training samples, three optimization
stages and 100000 evaluation samples per record. Stage 2 retains the original
adaptive variance stop and maximum six blocks. Each system emits 10 checkpoints
times two evaluation repeats. The unscored RMS diagnostics are skipped to save
compute; the scored energies and color observables use the original code.

The weaker variant keeps the architecture and Hamiltonian and reduces training
to 20 + 20 + 10*2 iterations. The more-training candidate warm starts from the
best *train-slice* checkpoint of reference seed 0, continues for 1000 iterations
at learning rate 0.0001, and evaluates its final ten checkpoints with the same
sampling settings. It is an actual optimization experiment; it is not assumed
to improve until its results demonstrate that.

Ignored `_backup/` stores a reversible copy of the initial checkpoint tree;
ignored `work/` stores generated training checkpoints and logs. The initial
tracked-file status is recorded in `initial_status.txt` because system-provided
README/manifest device files and trajectory removals predate this audit.
