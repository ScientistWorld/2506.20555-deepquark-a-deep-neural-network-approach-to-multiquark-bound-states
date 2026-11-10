#!/bin/bash
# Hardening job: retrain the reference (scripts/reproduce.sh, seed 0), score it with
# the new evaluators, evaluate earlier/variant states, shortcuts and noise.
# Sequential and resumable; see scripts/hardening/driver.py.
set -euo pipefail

cd /home/user
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# jax.export serialization needs flatbuffers (added to environment/setup.sh). If this
# job's container predates that, use the pure-Python copy unpacked from the PyPI wheel
# (flatbuffers 25.2.10, sha256 ebba5f4d...0051) into tmp/pydeps on the login node.
if ! python -c "import flatbuffers" 2>/dev/null; then
  echo "flatbuffers missing in the container; using tmp/pydeps for this job"
  export PYTHONPATH="/home/user/tmp/pydeps${PYTHONPATH:+:$PYTHONPATH}"
fi
python -c "import flatbuffers; print('flatbuffers', flatbuffers.__version__ if hasattr(flatbuffers, '__version__') else '?')"

python scripts/hardening/driver.py
