#!/usr/bin/env python3
"""Audit trials; mutations are confined to the documented submission artifacts.

No shortcut reads the hidden evaluator, hidden scores, or ranking reference.
Only the reporting helper reads the audit's frozen ranking specification.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/home/user')
RT = ROOT / 'redteam'
ART = ROOT / 'checkpoints'
SYSTEMS = {'ccqq': 'tcc_al1_binding', 'bbqq': 'tbb_al1_binding'}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=True) + '\n')


def restore_artifacts():
    if ART.exists():
        shutil.rmtree(ART)
    shutil.copytree(RT / '_backup/checkpoints', ART)


def command(args, cwd=ROOT, env=None, timeout=5400):
    print('RUN', ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, env=env, check=True, timeout=timeout)


def score(name, run_train=True):
    # Each script is an ordinary separate process. No monkeypatches, PYTHONPATH
    # changes, background processes, evaluator edits, or environment interception.
    if run_train:
        command(['bash', 'scripts/evaluate_train.sh'])
    command(['bash', 'scripts/evaluate_test.sh'])
    out = RT / name
    for split in ('train', 'test'):
        shutil.copy2(ROOT / f'scoring/scores_{split}.json', out / f'scores_{split}.json')
    (out / 'artifacts').mkdir(exist_ok=True)
    for system in SYSTEMS:
        source = ART / f'deepquark_{system}'
        if source.exists():
            shutil.copytree(source, out / 'artifacts' / source.name, dirs_exist_ok=True)
    report_rank(name)


def report_rank(name):
    # Audit reporting ONLY; never used to construct a shortcut's submission.
    spec = read(RT / 'reference/ranking_reference.json')['experiments']
    reference = read(RT / 'reference/original_scores_test.json')['experiments']
    actual = read(RT / name / 'scores_test.json')['experiments']
    detail = {}
    total = 0.0
    for exp, ref in reference.items():
        trial = actual[exp]['results']['deepquark']
        base = ref['results']['deepquark']
        terms = {key: spec[exp]['weight'] * cfg['coefficient'] * (trial[key] - base[key])
                 for key, cfg in spec[exp]['metrics'].items() if cfg['coefficient'] != 0}
        detail[exp] = terms
        total += sum(terms.values())
    write(RT / name / 'rank.json', {'delta_vs_shipped_test_reference': total, 'terms': detail})
    print('RANK', name, total, flush=True)


def train_config():
    # This is agent-visible code, unlike eval/test or reference scores.
    import runpy
    return runpy.run_path(str(ROOT / 'eval/train/evaluate.py'))['EXPERIMENTS']


def put_records(system, records, params):
    dest = ART / f'deepquark_{system}'
    dest.mkdir(parents=True, exist_ok=True)
    write(dest / 'results.json', records)
    write(dest / 'parameters.json', params)


def shortcut(name):
    configs = train_config()
    for system, exp in SYSTEMS.items():
        dest = ART / f'deepquark_{system}'
        if name == 'shortcut_visible_answer_replay':
            # Rebuild the test baseline answer from public train constants.
            # No wavefunction, optimizer, labels, or hidden reference is used.
            e = configs[exp]['threshold_mev'] / 1000
            put_records(system, {'0': {'1': {'Emean': e, 'Esigma': 0, 'color': '3*3:0,6*6:0'},
                                       '2': {'Emean': e, 'Esigma': 0, 'color': '3*3:0,6*6:0'}}},
                        {'hyperparams': {'para_num': 0}})
        elif name == 'shortcut_fabricated_energy':
            e = configs[exp]['threshold_mev'] / 1000 - 1
            put_records(system, {'0': {'1': {'Emean': e, 'Esigma': 0, 'color': '3*3:0.5,6*6:0.5'},
                                       '2': {'Emean': e, 'Esigma': 0, 'color': '3*3:0.5,6*6:0.5'}}},
                        {'hyperparams': {'para_num': 1857}})
        elif name == 'shortcut_impossible_values':
            put_records(system, {'0': {'1': {'Emean': -1e6, 'Esigma': -1e6, 'color': '3*3:9,6*6:9'},
                                       '2': {'Emean': -1e6, 'Esigma': -1e6, 'color': '3*3:9,6*6:9'}}},
                        {'hyperparams': {'para_num': -1e9}})
        elif name == 'shortcut_zero_metadata':
            # Reuse a trained reference's energies; only edit claimed costs.
            data = read(dest / 'results.json')
            params = read(dest / 'parameters.json')
            for repeats in data.values():
                for rec in repeats.values():
                    rec['Esigma'] = 0
            params['hyperparams']['para_num'] = 0
            put_records(system, data, params)
        elif name == 'shortcut_test_record_replay':
            # All records are method-generated, including the supposed holdout.
            # Replay the best record as both train and test, without inference.
            data = read(dest / 'results.json')
            best = min((rec for block in data.values() for rec in block.values()),
                       key=lambda rec: float(rec['Emean']))
            put_records(system, {'0': {'1': best, '2': best}}, read(dest / 'parameters.json'))
        elif name == 'shortcut_changed_hamiltonian':
            # Exact evaluation under H' = H - 0.1 GeV * identity: the same
            # trained state and variance, but the wrong fixed task. This is
            # mathematically equivalent to changing the local-energy operator.
            data = read(dest / 'results.json')
            params = read(dest / 'parameters.json')
            for repeats in data.values():
                for rec in repeats.values():
                    rec['Emean'] = float(rec['Emean']) - 0.1
            params['hamiltonian_identity_shift_gev'] = -0.1
            put_records(system, data, params)
        elif name == 'shortcut_threshold_change':
            params = read(dest / 'parameters.json')
            params['threshold_mev'] = 1e9
            params['ground_truth_options'] = {'threshold_mev': 1e9}
            write(dest / 'parameters.json', params)
        elif name == 'shortcut_null_records':
            put_records(system, {'0': {'1': {'Emean': None}, '2': {'Emean': None}}},
                        {'hyperparams': {'para_num': 0}})
        elif name == 'shortcut_empty_records':
            put_records(system, {}, {'hyperparams': {'para_num': 0}})
        elif name == 'shortcut_direct_scores':
            pass
        else:
            raise ValueError(name)

    if name == 'shortcut_direct_scores':
        # Write spectacular scalar answers, then let the genuine test replace
        # them. The train registry is valid, so this isolates scalar trust.
        command(['bash', 'scripts/evaluate_train.sh'])
        forged = read(ROOT / 'scoring/scores_train.json')
        for exp in SYSTEMS.values():
            for key in forged['experiments'][exp]['results']['deepquark']:
                forged['experiments'][exp]['results']['deepquark'][key] = -1e12
        write(ROOT / 'scoring/scores_train.json', forged)
        write(ROOT / 'scoring/scores_test.json', forged)
        score(name, run_train=False)
    else:
        # Invalid records must reach the hidden test too; the unchanged train
        # registry already names deepquark for both experiments.
        score(name, run_train=name not in ('shortcut_null_records', 'shortcut_empty_records'))


SEED_HELPER = '''
# Audit instrumentation: seed both parameter initialization and MC sampling.
_audit_counter = 0
def seeded_mcstate(*args, **kwargs):
    global _audit_counter
    _audit_counter += 1
    s = int(os.environ['RT_SEED']) + 1009 * _audit_counter
    return nk.vqs.MCState(*args, **kwargs, seed=s, sampler_seed=s + 500003)

'''


def make_work(name):
    work = RT / 'work' / name / 'DeepQuark'
    # A scheduler timeout must not erase completed continuation checkpoints.
    if name == 'better_more_training' and work.exists():
        return work
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(ROOT / 'method/DeepQuark', work,
                    ignore=shutil.ignore_patterns('AL1', '.git', '__pycache__'))
    for file in ('train.py', 'eval.py', 'continue_train3.py'):
        path = work / 'multiquark' / file
        source = path.read_text().replace('nk.vqs.MCState(', 'seeded_mcstate(')
        anchor = '"""Model"""' if file != 'continue_train3.py' else 'def env_int(name, default):'
        source = source.replace(anchor, SEED_HELPER + '\n' + anchor, 1)
        if name == 'honest_more_samples' and file == 'eval.py':
            # More samples need not enlarge the local-energy working batch.
            source = source.replace('chunk_size=max(1, nsample // 4)',
                                    'chunk_size=min(25000, max(1, nsample // 4))')
        path.write_text(source)
    return work


def continuation(work, system, args, env):
    source = RT / 'work/reference_seed_0/DeepQuark/AL1' / f'{system}(S=1.0_I=0.0)' / '1'
    dest = work / 'AL1' / f'{system}(S=1.0_I=0.0)' / '1'
    if not dest.exists():
        shutil.copytree(source, dest)
    # Select the initial state using ONLY the public train slice.
    records = read(source / 'results.json')
    items = [(ck, rec) for ck, values in sorted(records.items())
             for _, rec in sorted(values.items())]
    best_ck, _ = min(items[::2], key=lambda item: float(item[1]['Emean']))
    complete = []
    discarded = []
    for path in dest.glob('Train3_*.mpack'):
        index = int(path.stem.removeprefix('Train3_'))
        if index < 10:
            continue
        logpath = path.with_suffix('.log')
        # NetKet writes snapshots during a block too. Only retain full blocks;
        # the interrupted 17-step block is restarted from the previous state.
        steps = read(logpath)['Energy']['iters'] if logpath.exists() else []
        if len(steps) == 50 and steps[-1] % 50 == 49:
            complete.append(index)
        else:
            discarded.append(index)
            path.unlink()
            if logpath.exists():
                logpath.unlink()
    first_new = max(complete, default=9) + 1
    start = str(first_new - 1) if complete else str(best_ck)
    plan = {'initial_train_selected_checkpoint': str(best_ck),
            'resume_from': start, 'first_new_index': first_new,
            'blocks_remaining': max(0, 30 - first_new),
            'completed_extra_blocks_before_resume': sorted(complete),
            'discarded_partial_blocks': sorted(discarded),
            'target_extra_blocks': 20, 'iterations_per_block': 50, 'lr': 0.0001}
    history_path = RT / 'better_more_training' / f'resume_history_{system}.json'
    history = read(history_path) if history_path.exists() else []
    history.append(plan)
    write(history_path, history)
    if first_new < 30:
        env.update({'DQ_RESUME_FROM': start, 'DQ_RESUME_START_INDEX': str(first_new),
                    'DQ_RESUME_TRAIN3_BLOCKS': str(30 - first_new),
                    'DQ_RESUME_TRAIN3_ITERS': '50', 'DQ_RESUME_LR': '0.0001'})
        command([sys.executable, 'multiquark/continue_train3.py', *args, '--n', '1'], work, env)


def quality_variant(name):
    seed = int(name.rsplit('_', 1)[-1]) if name.startswith('reference_seed_') else 0
    env = os.environ.copy()
    env.update({
        'JAX_PLATFORM_NAME': 'gpu', 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false', 'RT_SEED': str(seed),
        'DQ_NCHAINS': '2000', 'DQ_NSWEEP': '20', 'DQ_NSAMPLE': '20000', 'DQ_NDISCARD': '100',
        'DQ_TRAIN1_ITERS': '1000', 'DQ_TRAIN2_BLOCKS': '6', 'DQ_TRAIN2_ITERS': '500',
        'DQ_TRAIN2_NDISCARD': '200', 'DQ_CHECK_NCHAINS': '2000', 'DQ_CHECK_NSWEEP': '50',
        'DQ_CHECK_NSAMPLE': '100000', 'DQ_CHECK_NDISCARD': '500', 'DQ_CHECK_REPEATS': '2',
        'DQ_VARIANCE_TARGET': '0.002', 'DQ_TRAIN3_BLOCKS': '10', 'DQ_TRAIN3_ITERS': '50',
        'DQ_EVAL_NCHAINS': '2000', 'DQ_EVAL_NSWEEP': '100', 'DQ_EVAL_NSAMPLE': '100000',
        'DQ_EVAL_NDISCARD': '1000', 'DQ_EVAL_REPEATS': '2', 'DQ_EVAL_SKIP_RMS': '1',
    })
    if name == 'weaker_less_training':
        env.update({'DQ_TRAIN1_ITERS': '20', 'DQ_TRAIN2_BLOCKS': '1', 'DQ_TRAIN2_ITERS': '20',
                    'DQ_TRAIN3_ITERS': '2'})
    if name == 'honest_more_samples':
        env['DQ_EVAL_NSAMPLE'] = '400000'
    work = make_work(name)
    for system in SYSTEMS:
        args = ['AL1', system, '1', '0']
        saved = RT / name / 'artifacts' / f'deepquark_{system}'
        if name == 'better_more_training' and all((saved / f).exists() for f in
                ('parameters.json', 'train.json', 'eval.json', 'results.json')):
            print('REUSE completed system', name, system, flush=True)
            shutil.copytree(saved, ART / f'deepquark_{system}', dirs_exist_ok=True)
            continue
        if name == 'better_more_training':
            continuation(work, system, args, env)
            indices = list(range(20, 30))
        elif name == 'honest_more_samples':
            source = RT / 'work/reference_seed_0/DeepQuark/AL1' / f'{system}(S=1.0_I=0.0)' / '1'
            dest = work / 'AL1' / f'{system}(S=1.0_I=0.0)' / '1'
            shutil.copytree(source, dest, dirs_exist_ok=True)
            indices = list(range(10))
        else:
            command([sys.executable, 'multiquark/train.py', *args], work, env)
            indices = list(range(10))
        command([sys.executable, 'multiquark/eval.py', *args, '--n', '1', '--irange', str(indices)], work, env)
        source = work / 'AL1' / f'{system}(S=1.0_I=0.0)' / '1'
        for file in ('parameters.json', 'train.json', 'eval.json', 'results.json'):
            shutil.copy2(source / file, ART / f'deepquark_{system}' / file)
        # Save completed systems immediately, even if the next process fails.
        shutil.copytree(ART / f'deepquark_{system}', RT / name / 'artifacts' / f'deepquark_{system}', dirs_exist_ok=True)
    score(name)


def main():
    name = sys.argv[1]
    out = RT / name
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    restore_artifacts()
    try:
        if name == 'reference':
            score(name)
        elif name.startswith('shortcut_'):
            shortcut(name)
        else:
            quality_variant(name)
        write(out / 'status.json', {'status': 'completed', 'seconds': time.time() - start})
    except Exception as exc:
        write(out / 'status.json', {'status': 'failed', 'exception': repr(exc), 'seconds': time.time() - start})
        raise
    finally:
        restore_artifacts()
        # Keep all tracked score files exactly as they were at audit start.
        start_commit = (RT / 'start_commit.txt').read_text().strip()
        subprocess.run(['git', 'restore', '--source', start_commit, '--',
                        'scoring/scores_train.json', 'scoring/scores_test.json'], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
