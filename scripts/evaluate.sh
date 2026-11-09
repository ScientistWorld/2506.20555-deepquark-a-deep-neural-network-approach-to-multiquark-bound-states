#!/bin/bash
# Score every submission under checkpoints/ with the public evaluation protocol
# and validate the workspace.
set -euo pipefail

cd /home/user

python eval/train/evaluate.py --artifacts checkpoints --output scoring/scores.json
python validate.py --compare
