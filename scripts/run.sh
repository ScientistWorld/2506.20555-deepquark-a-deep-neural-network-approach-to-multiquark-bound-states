#!/bin/bash
# Hardening job: retrain the reference (scripts/reproduce.sh, seed 0), train the
# better/seed variants, build shortcut submissions and score everything with
# the new evaluators. Resumable; see scripts/hardening/driver.py.
set -euo pipefail

cd /home/user
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python scripts/hardening/driver.py
