#!/bin/bash
# Reproduce the reference end to end: train all three systems with the authors'
# released DeepQuark schedule (DQ_SEED=0), export the selected states to
# checkpoints/deepquark_*/wavefunction.jaxexport, and score them with the train
# and test evaluators (scripts/reproduce.sh).
set -euo pipefail

cd /home/user
export XLA_PYTHON_CLIENT_PREALLOCATE=false

bash scripts/reproduce.sh
