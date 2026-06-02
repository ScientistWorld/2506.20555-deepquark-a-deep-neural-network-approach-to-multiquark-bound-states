#!/bin/bash
set -euo pipefail

cd /home/user/method/DeepQuark

POTENTIAL="${DQ_POTENTIAL:-AL1}"
QUARKS="${DQ_QUARKS:-ccqq}"
SPIN="${DQ_SPIN:-1}"
ISOSPIN="${DQ_ISOSPIN:-0}"
ARTIFACT_NAME="${DQ_ARTIFACT_NAME:-deepquark_${QUARKS}}"

python multiquark/train.py "$POTENTIAL" "$QUARKS" "$SPIN" "$ISOSPIN"

RUN_PARENT="$POTENTIAL/${QUARKS}(S=${SPIN}.0_I=${ISOSPIN}.0)"
if [ ! -d "$RUN_PARENT" ]; then
  RUN_PARENT="$POTENTIAL/${QUARKS}(S=${SPIN}_I=${ISOSPIN})"
fi
RUN_ID="$(find "$RUN_PARENT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -n | tail -1)"

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
