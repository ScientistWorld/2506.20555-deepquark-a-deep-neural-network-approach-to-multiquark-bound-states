#!/bin/bash
set -euo pipefail

cd /home/user

python eval/deepquark_scores.py --artifacts checkpoints --output scoring/scores.json
python validate.py --compare
