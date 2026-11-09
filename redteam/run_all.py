#!/usr/bin/env python3
"""One compute job; independent trials checkpoint results as they finish."""
import json
from pathlib import Path
import subprocess
import sys

root = Path('/home/user')
rt = root / 'redteam'
names = [
    'reference', 'shortcut_visible_answer_replay', 'shortcut_fabricated_energy',
    'shortcut_impossible_values', 'shortcut_zero_metadata', 'shortcut_test_record_replay',
    'shortcut_changed_hamiltonian', 'shortcut_threshold_change', 'shortcut_direct_scores',
    'shortcut_null_records', 'shortcut_empty_records',
    'weaker_less_training', 'reference_seed_0', 'reference_seed_1', 'reference_seed_2',
    'better_more_training', 'honest_more_samples',
]
for name in names:
    folder = rt / name
    status = folder / 'status.json'
    if status.exists() and json.loads(status.read_text()).get('status') == 'completed':
        print('SKIP completed', name, flush=True)
        continue
    print('START', name, flush=True)
    with (folder / 'run.log').open('a') as log:
        result = subprocess.run(['bash', str(folder / 'run.sh')], cwd=root, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        lines = (folder / 'run.log').read_text().splitlines()
        (folder / 'failure_output.txt').write_text('\n'.join(lines[-50:]) + '\n')
    print('FINISHED', name, 'exit', result.returncode, flush=True)
