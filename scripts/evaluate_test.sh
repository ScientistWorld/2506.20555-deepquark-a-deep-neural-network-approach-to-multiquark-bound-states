#!/bin/bash
# Held-out test scoring of every method found under checkpoints/ (no arguments).
set -euo pipefail

cd /home/user

python eval/test/evaluate.py --artifacts checkpoints --output scoring/scores_test.json
