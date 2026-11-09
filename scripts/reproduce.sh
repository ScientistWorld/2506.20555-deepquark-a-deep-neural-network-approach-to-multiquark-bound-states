#!/bin/bash
# Reproduce the reference DeepQuark submissions for all scored systems from
# scratch, then score them.  Settings are the authors' released defaults
# (train.py / eval.py), with a fixed seed so the run is repeatable.
set -euo pipefail

cd /home/user

export JAX_PLATFORM_NAME="${JAX_PLATFORM_NAME:-gpu}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

run_target() {
  local quarks="$1"
  local spin="$2"
  local artifact_name="$3"

  (
    export DQ_POTENTIAL=AL1
    export DQ_QUARKS="$quarks"
    export DQ_SPIN="$spin"
    export DQ_ISOSPIN=0
    export DQ_ARTIFACT_NAME="$artifact_name"
    export DQ_SEED="${DQ_SEED:-0}"

    export DQ_NCHAINS="${DQ_NCHAINS:-2000}"
    export DQ_NSWEEP="${DQ_NSWEEP:-20}"
    export DQ_NSAMPLE="${DQ_NSAMPLE:-20000}"
    export DQ_NDISCARD="${DQ_NDISCARD:-100}"
    export DQ_TRAIN1_ITERS="${DQ_TRAIN1_ITERS:-1000}"
    export DQ_TRAIN2_BLOCKS="${DQ_TRAIN2_BLOCKS:-10}"
    export DQ_TRAIN2_ITERS="${DQ_TRAIN2_ITERS:-500}"
    export DQ_TRAIN2_NDISCARD="${DQ_TRAIN2_NDISCARD:-200}"
    export DQ_CHECK_NCHAINS="${DQ_CHECK_NCHAINS:-2000}"
    export DQ_CHECK_NSWEEP="${DQ_CHECK_NSWEEP:-50}"
    export DQ_CHECK_NSAMPLE="${DQ_CHECK_NSAMPLE:-400000}"
    export DQ_CHECK_NDISCARD="${DQ_CHECK_NDISCARD:-500}"
    export DQ_CHECK_REPEATS="${DQ_CHECK_REPEATS:-5}"
    export DQ_VARIANCE_TARGET="${DQ_VARIANCE_TARGET:-0.002}"
    export DQ_TRAIN3_BLOCKS="${DQ_TRAIN3_BLOCKS:-10}"
    export DQ_TRAIN3_ITERS="${DQ_TRAIN3_ITERS:-50}"
    export DQ_EVAL_NCHAINS="${DQ_EVAL_NCHAINS:-2000}"
    export DQ_EVAL_NSWEEP="${DQ_EVAL_NSWEEP:-100}"
    export DQ_EVAL_NSAMPLE="${DQ_EVAL_NSAMPLE:-400000}"
    export DQ_EVAL_NDISCARD="${DQ_EVAL_NDISCARD:-1000}"
    export DQ_EVAL_REPEATS="${DQ_EVAL_REPEATS:-2}"
    # rms radii are diagnostics that the gym does not score
    export DQ_EVAL_SKIP_RMS="${DQ_EVAL_SKIP_RMS:-1}"

    unset DQ_SKIP_TRAIN DQ_RUN_ID DQ_EVAL_IRANGE DQ_SKIP_EVAL DQ_EXPORT_INDEX
    bash scripts/method.sh
  )
}

for system in ${DQ_SYSTEMS:-ccqq bbqq ccqqc}; do
  case "$system" in
    ccqq) run_target ccqq 1 deepquark_ccqq ;;
    bbqq) run_target bbqq 1 deepquark_bbqq ;;
    ccqqc) run_target ccqqc 2.5 deepquark_ccqqc ;;
    *) echo "unknown system $system" >&2; exit 1 ;;
  esac
done

if [ "${DQ_SKIP_EVALUATE:-0}" != "1" ]; then
  bash scripts/evaluate_train.sh
  bash scripts/evaluate_test.sh
  bash scripts/evaluate.sh
fi
