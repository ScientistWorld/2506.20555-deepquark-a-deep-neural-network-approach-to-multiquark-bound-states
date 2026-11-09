#!/bin/bash
# Public train-slice scoring of every method found under checkpoints/.
set -euo pipefail

cd /home/user

python eval/train/evaluate.py --artifacts checkpoints --output scoring/scores_train.json
