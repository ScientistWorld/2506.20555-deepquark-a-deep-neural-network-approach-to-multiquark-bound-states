#!/bin/bash
set -euo pipefail

cd /home/user

python eval/train/evaluate.py --artifacts checkpoints --output scoring/scores_train.json --experiments tcc_al1_binding,tbb_al1_binding
