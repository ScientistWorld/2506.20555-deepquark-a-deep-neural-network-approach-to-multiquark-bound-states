#!/bin/bash
# Train (optional), evaluate and export one DeepQuark system.
# The exported wavefunction lands in checkpoints/$ARTIFACT_NAME/wavefunction.jaxexport,
# which is the only file the gym evaluators read.
set -euo pipefail

ROOT="${DQ_ROOT:-/home/user}"
cd "${DQ_CODE_DIR:-$ROOT/method/DeepQuark}"

POTENTIAL="${DQ_POTENTIAL:-AL1}"
QUARKS="${DQ_QUARKS:-ccqq}"
SPIN="${DQ_SPIN:-1}"
ISOSPIN="${DQ_ISOSPIN:-0}"
ARTIFACT_NAME="${DQ_ARTIFACT_NAME:-deepquark_${QUARKS}}"
DEST="${DQ_ARTIFACT_ROOT:-$ROOT/checkpoints}/$ARTIFACT_NAME"

if [ "${DQ_SKIP_TRAIN:-0}" != "1" ]; then
  python multiquark/train.py "$POTENTIAL" "$QUARKS" "$SPIN" "$ISOSPIN"
else
  echo "Skipping training because DQ_SKIP_TRAIN=1"
fi

RUN_PARENT="$(python - "$POTENTIAL" "$QUARKS" "$SPIN" "$ISOSPIN" <<'PY'
import sys

potential, quarks, spin, isospin = sys.argv[1:]
print(f"{potential}/{quarks}(S={float(spin)}_I={float(isospin)})")
PY
)"
if [ ! -d "$RUN_PARENT" ]; then
  echo "Expected DeepQuark run parent is missing: $RUN_PARENT" >&2
  exit 1
fi

if [ -n "${DQ_RUN_ID:-}" ]; then
  RUN_ID="$DQ_RUN_ID"
else
  RUN_ID="$(find "$RUN_PARENT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -n | tail -1)"
fi
if [ -z "$RUN_ID" ] || [ ! -d "$RUN_PARENT/$RUN_ID" ]; then
  echo "Expected DeepQuark run directory is missing: $RUN_PARENT/$RUN_ID" >&2
  exit 1
fi

# Optional refinement of the final stage at an annealed learning rate (used for the
# pentaquark, whose final stage at lr 1e-3 hovers at the threshold): DQ_REFINE_BLOCKS more
# blocks of DQ_TRAIN3_ITERS SR iterations at DQ_REFINE_LR, continuing from the last final-stage
# checkpoint (DQ_REFINE_FROM overrides) and saved as further Train3_* checkpoints.
TRAIN3_BLOCKS="${DQ_TRAIN3_BLOCKS:-10}"
REFINE_BLOCKS="${DQ_REFINE_BLOCKS:-0}"
if [ "$REFINE_BLOCKS" != "0" ] && [ "${DQ_SKIP_TRAIN:-0}" != "1" ]; then
  DQ_RESUME_FROM="${DQ_REFINE_FROM:-$((TRAIN3_BLOCKS - 1))}" \
  DQ_RESUME_START_INDEX="$TRAIN3_BLOCKS" \
  DQ_RESUME_TRAIN3_BLOCKS="$REFINE_BLOCKS" \
  DQ_RESUME_TRAIN3_ITERS="${DQ_TRAIN3_ITERS:-50}" \
  DQ_RESUME_LR="${DQ_REFINE_LR:-0.0001}" \
    python multiquark/continue_train3.py "$POTENTIAL" "$QUARKS" "$SPIN" "$ISOSPIN" --n "$RUN_ID"
fi

if [ -n "${DQ_EVAL_IRANGE:-}" ]; then
  IRANGE="$DQ_EVAL_IRANGE"
else
  IRANGE_END=$((TRAIN3_BLOCKS + REFINE_BLOCKS))
  IRANGE="$(python -c "print(list(range($IRANGE_END)))")"
fi
if [ "${DQ_SKIP_EVAL:-0}" != "1" ]; then
  python multiquark/eval.py "$POTENTIAL" "$QUARKS" "$SPIN" "$ISOSPIN" --n "$RUN_ID" --irange "$IRANGE"
fi

# Select the lowest-energy checkpoint of the method's own evaluation and export it.
mkdir -p "$DEST"
DQ_EXPORT_DIR="$DEST" python multiquark/export.py "$POTENTIAL" "$QUARKS" "$SPIN" "$ISOSPIN" \
  --n "$RUN_ID" --irange "$IRANGE"

# Provenance only; the evaluators never read these files.
for f in parameters.json train.json eval.json results.json; do
  if [ -f "$RUN_PARENT/$RUN_ID/$f" ]; then
    cp "$RUN_PARENT/$RUN_ID/$f" "$DEST/$f"
  fi
done
