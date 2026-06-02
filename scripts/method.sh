#!/bin/bash
set -euo pipefail

cd /home/user/method/DeepQuark

POTENTIAL="${DQ_POTENTIAL:-AL1}"
QUARKS="${DQ_QUARKS:-ccqq}"
SPIN="${DQ_SPIN:-1}"
ISOSPIN="${DQ_ISOSPIN:-0}"
ARTIFACT_NAME="${DQ_ARTIFACT_NAME:-deepquark_${QUARKS}}"

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

IRANGE_END="${DQ_TRAIN3_BLOCKS:-10}"
IRANGE_END=$((IRANGE_END - 1))
IRANGE="$(python - <<PY
print(list(range($IRANGE_END + 1)))
PY
)"
python multiquark/eval.py "$POTENTIAL" "$QUARKS" "$SPIN" "$ISOSPIN" --n "$RUN_ID" --irange "$IRANGE"

DEST="/home/user/checkpoints/$ARTIFACT_NAME"
mkdir -p "$DEST"
cp "$RUN_PARENT/$RUN_ID/parameters.json" "$DEST/parameters.json"
cp "$RUN_PARENT/$RUN_ID/train.json" "$DEST/train.json"
cp "$RUN_PARENT/$RUN_ID/eval.json" "$DEST/eval.json"
cp "$RUN_PARENT/$RUN_ID/results.json" "$DEST/results.json"
