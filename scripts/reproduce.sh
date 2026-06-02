#!/bin/bash
set -euo pipefail

cd /home/user

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

    # Reproduce the accepted score set with the same scale used by the
    # successful GPU jobs. This preserves the DeepQuark procedure while
    # keeping the end-to-end script feasible under the benchmark budget.
    export DQ_NCHAINS="${DQ_NCHAINS:-2000}"
    export DQ_NSWEEP="${DQ_NSWEEP:-20}"
    export DQ_NSAMPLE="${DQ_NSAMPLE:-20000}"
    export DQ_NDISCARD="${DQ_NDISCARD:-100}"
    export DQ_TRAIN1_ITERS="${DQ_TRAIN1_ITERS:-1000}"
    export DQ_TRAIN2_BLOCKS="${DQ_TRAIN2_BLOCKS:-6}"
    export DQ_TRAIN2_ITERS="${DQ_TRAIN2_ITERS:-500}"
    export DQ_TRAIN2_NDISCARD="${DQ_TRAIN2_NDISCARD:-200}"
    export DQ_CHECK_NCHAINS="${DQ_CHECK_NCHAINS:-2000}"
    export DQ_CHECK_NSWEEP="${DQ_CHECK_NSWEEP:-50}"
    export DQ_CHECK_NSAMPLE="${DQ_CHECK_NSAMPLE:-100000}"
    export DQ_CHECK_NDISCARD="${DQ_CHECK_NDISCARD:-500}"
    export DQ_CHECK_REPEATS="${DQ_CHECK_REPEATS:-2}"
    export DQ_VARIANCE_TARGET="${DQ_VARIANCE_TARGET:-0.002}"
    export DQ_TRAIN3_BLOCKS="${DQ_TRAIN3_BLOCKS:-10}"
    export DQ_TRAIN3_ITERS="${DQ_TRAIN3_ITERS:-50}"
    export DQ_EVAL_NCHAINS="${DQ_EVAL_NCHAINS:-2000}"
    export DQ_EVAL_NSWEEP="${DQ_EVAL_NSWEEP:-100}"
    export DQ_EVAL_NSAMPLE="${DQ_EVAL_NSAMPLE:-100000}"
    export DQ_EVAL_NDISCARD="${DQ_EVAL_NDISCARD:-1000}"
    export DQ_EVAL_REPEATS="${DQ_EVAL_REPEATS:-2}"

    unset DQ_SKIP_TRAIN DQ_RUN_ID DQ_EVAL_IRANGE DQ_EVAL_SKIP_RMS
    bash scripts/method.sh
  )
}

run_target ccqq 1 deepquark_ccqq
run_target bbqq 1 deepquark_bbqq
bash scripts/evaluate.sh
