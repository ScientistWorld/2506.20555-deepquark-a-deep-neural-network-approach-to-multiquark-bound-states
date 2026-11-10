#!/bin/bash
# Final hardening job: score the reference, variants and shortcuts with the final
# evaluators and train one time-boxed "better" variant (scripts/hardening/final_job.py).
# After the job this file is restored to the end-to-end reference reproduction.
set -euo pipefail

cd /home/user
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# jax.export serialization needs flatbuffers (environment/setup.sh installs it).
if ! python -c "import flatbuffers" 2>/dev/null; then
  echo "flatbuffers missing in the container; using tmp/pydeps for this job"
  export PYTHONPATH="/home/user/tmp/pydeps${PYTHONPATH:+:$PYTHONPATH}"
fi

python scripts/hardening/final_job.py
