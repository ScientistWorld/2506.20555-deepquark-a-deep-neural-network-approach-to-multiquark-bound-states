#!/bin/bash
set -euo pipefail

cd /home/user

python eval/deepquark_scores.py --artifacts checkpoints --output scoring/scores.json --experiments tcc_al1_binding,tbb_al1_binding
python validate.py --compare
