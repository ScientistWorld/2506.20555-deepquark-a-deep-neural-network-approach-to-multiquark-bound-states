#!/usr/bin/env python3
"""Hardening job: retrain the reference, train variants, build shortcuts, score all.

Stages run one at a time (a job gets a single CPU; concurrent JAX processes slow
each other down by an order of magnitude) in priority order.  A stage starts only
if its estimated duration fits in the remaining job time; finished stages leave a
marker in tmp/hardening/state, so resubmitting the same job continues the plan.

Usage: python scripts/hardening/driver.py
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/user")
HW = ROOT / "tmp/hardening"
ART = HW / "artifacts"
WORK = HW / "work"
STATE = HW / "state"
LOGS = HW / "logs"
CHAINS = HW / "chains"
OUT = ROOT / "notes/hardening"
RES = OUT / "results"
METHOD = ROOT / "method/DeepQuark"
EXPORT = METHOD / "multiquark/export.py"
RT_WORK = ROOT / "redteam/work"
TEST_EVAL = ROOT / "eval/test/evaluate.py"
TRAIN_EVAL = ROOT / "eval/train/evaluate.py"
START = float(os.environ.get("HW_START", time.time()))
BUDGET = float(os.environ.get("HW_BUDGET_HOURS", "5.75")) * 3600

SPIN = {"ccqq": "1", "bbqq": "1", "ccqqc": "2.5"}
PARENT = {"ccqq": "AL1/ccqq(S=1.0_I=0.0)", "bbqq": "AL1/bbqq(S=1.0_I=0.0)", "ccqqc": "AL1/ccqqc(S=2.5_I=0.0)"}
TEN = "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]"

REF_ENV = {
    "DQ_SEED": "0", "DQ_NCHAINS": "2000", "DQ_NSWEEP": "20", "DQ_NSAMPLE": "20000", "DQ_NDISCARD": "100",
    "DQ_TRAIN1_ITERS": "1000", "DQ_TRAIN2_BLOCKS": "10", "DQ_TRAIN2_ITERS": "500", "DQ_TRAIN2_NDISCARD": "200",
    "DQ_CHECK_NCHAINS": "2000", "DQ_CHECK_NSWEEP": "50", "DQ_CHECK_NSAMPLE": "400000",
    "DQ_CHECK_NDISCARD": "500", "DQ_CHECK_REPEATS": "5", "DQ_VARIANCE_TARGET": "0.002",
    "DQ_TRAIN3_BLOCKS": "10", "DQ_TRAIN3_ITERS": "50", "DQ_EVAL_NCHAINS": "2000", "DQ_EVAL_NSWEEP": "100",
    "DQ_EVAL_NSAMPLE": "400000", "DQ_EVAL_NDISCARD": "1000", "DQ_EVAL_REPEATS": "2", "DQ_EVAL_SKIP_RMS": "1",
}
BETTER_ENV = dict(REF_ENV, DQ_VARIANCE_TARGET="0", DQ_CHECK_REPEATS="1", DQ_CHECK_NSAMPLE="100000",
                  DQ_TRAIN3_NSAMPLE="100000")

for d in (ART, WORK, STATE, LOGS, CHAINS, RES):
    d.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - START) / 60:5.1f}m] {msg}", flush=True)


def gpu_count():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=60).stdout
        return max(1, sum(1 for line in out.splitlines() if line.startswith("GPU")))
    except Exception:  # noqa: BLE001
        return 0


def dev_env(dev):
    return {"JAX_CUDA_VISIBLE_DEVICES": str(dev), "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}


def run(cmd, cwd=ROOT, env=None, logname=None):
    full = os.environ.copy()
    full.update(env or {})
    log(f"RUN {' '.join(map(str, cmd))}  (cwd={cwd})")
    with open(LOGS / f"{logname or 'misc'}.log", "a") as handle:
        handle.write(f"\n===== {time.ctime()} {' '.join(map(str, cmd))}\n")
        handle.flush()
        proc = subprocess.run(list(map(str, cmd)), cwd=cwd, env=full, stdout=handle, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        tail = (LOGS / f"{logname or 'misc'}.log").read_text()[-3000:]
        raise RuntimeError(f"rc={proc.returncode} for {cmd}\n{tail}")


def stage(name, fn, optional_after=None):
    marker = STATE / f"{name}.done"
    if marker.exists():
        log(f"skip {name} (done)")
        return True
    if optional_after is not None and time.time() - START > optional_after:
        log(f"skip {name} (time budget)")
        return False
    log(f"start {name}")
    t = time.time()
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        log(f"FAILED {name}: {exc!r}"[:4000])
        (STATE / f"{name}.failed").write_text(repr(exc)[:20000])
        return False
    marker.write_text(json.dumps({"seconds": time.time() - t}))
    (STATE / f"{name}.failed").unlink(missing_ok=True)
    log(f"done {name} in {time.time() - t:.0f}s")
    return True


def wait_for(names, limit):
    while time.time() - START < limit:
        if all((STATE / f"{n}.done").exists() or (STATE / f"{n}.failed").exists() for n in names):
            return
        time.sleep(60)
    log(f"wait_for timed out: {names}")


# ---------------------------------------------------------------- training
def run_ids(code_dir, system, need_train_json=False):
    parent = Path(code_dir) / PARENT[system]
    if not parent.is_dir():
        return []
    return sorted(int(d.name) for d in parent.iterdir()
                  if d.is_dir() and d.name.isdigit() and (not need_train_json or (d / "train.json").exists()))


def base_run(code_dir, system):
    """Largest run id that existed before this hardening job started training (recorded once)."""
    path = STATE / "base_runs.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    key = f"{code_dir}:{system}"
    if key not in data:
        ids = run_ids(code_dir, system)
        data[key] = max(ids) if ids else 0
        path.write_text(json.dumps(data, indent=2))
    return data[key]


def resume_env(code_dir, system):
    """If this job already finished training a run, skip retraining (and re-evaluation when complete)."""
    done = [r for r in run_ids(code_dir, system, need_train_json=True) if r > base_run(code_dir, system)]
    if not done:
        return {}
    rid = max(done)
    env = {"DQ_SKIP_TRAIN": "1", "DQ_RUN_ID": str(rid)}
    res = Path(code_dir) / PARENT[system] / str(rid) / "results.json"
    try:
        results = json.loads(res.read_text())
        if len(results) >= 10 and all(len(v) >= 2 for v in results.values()):
            env["DQ_SKIP_EVAL"] = "1"
    except (OSError, json.JSONDecodeError):
        pass
    log(f"resume {code_dir} {system} from run {rid}: {env}")
    return env


def train_variant(variant, system, env, dev):
    work = WORK / variant / "DeepQuark"
    if not work.exists():
        shutil.copytree(METHOD, work, ignore=shutil.ignore_patterns("AL1", "__pycache__"))
    full = dict(env)
    full.update(dev_env(dev))
    full.update({"JAX_PLATFORM_NAME": "gpu", "DQ_CODE_DIR": str(work), "DQ_POTENTIAL": "AL1",
                 "DQ_QUARKS": system, "DQ_SPIN": SPIN[system], "DQ_ISOSPIN": "0",
                 "DQ_ARTIFACT_NAME": f"deepquark_{system}", "DQ_ARTIFACT_ROOT": str(ART / variant)})
    full.update(resume_env(work, system))
    run(["bash", "scripts/method.sh"], env=full, logname=f"{variant}_{system}")


def ref_system(system, dev):
    resume = resume_env(METHOD, system)
    if not resume:
        run(["bash", "scripts/reproduce.sh"],
            env={**dev_env(dev), "DQ_SYSTEMS": system, "DQ_SKIP_EVALUATE": "1"}, logname=f"ref_{system}")
        return
    # training already finished in an earlier attempt: evaluate/export with the same settings
    env = {**REF_ENV, **dev_env(dev), **resume, "JAX_PLATFORM_NAME": "gpu", "DQ_POTENTIAL": "AL1",
           "DQ_QUARKS": system, "DQ_SPIN": SPIN[system], "DQ_ISOSPIN": "0",
           "DQ_ARTIFACT_NAME": f"deepquark_{system}"}
    run(["bash", "scripts/method.sh"], env=env, logname=f"ref_{system}")


# ---------------------------------------------------------------- evaluation
def export_state(code_dir, system, run_id, irange, variant, dev, extra=None):
    env = {**dev_env(dev), "DQ_EXPORT_DIR": str(ART / variant / f"deepquark_{system}"), **(extra or {})}
    run([sys.executable, EXPORT, "AL1", system, SPIN[system], "0", "--n", run_id, "--irange", irange],
        cwd=code_dir, env=env, logname=f"export_{variant}")


def old_pcc_reference(dev):
    """Standard-pipeline Train3_0..9 of the previous pentaquark run, evaluated as before."""
    work = WORK / "old_pcc" / "DeepQuark"
    if not work.exists():
        shutil.copytree(METHOD, work, ignore=shutil.ignore_patterns("AL1", "__pycache__"))
    src = METHOD / PARENT["ccqqc"] / "1"
    dst = work / PARENT["ccqqc"] / "1"
    dst.mkdir(parents=True, exist_ok=True)
    for f in [f"Train3_{i}.mpack" for i in range(10)] + ["parameters.json", "train.json"]:
        shutil.copy2(src / f, dst / f)
    env = {**dev_env(dev), "JAX_PLATFORM_NAME": "gpu", "DQ_EVAL_NCHAINS": "2000", "DQ_EVAL_NSWEEP": "100",
           "DQ_EVAL_NSAMPLE": "100000", "DQ_EVAL_NDISCARD": "1000", "DQ_EVAL_REPEATS": "2",
           "DQ_EVAL_SKIP_RMS": "1", "DQ_SEED": "0"}
    run([sys.executable, "multiquark/eval.py", "AL1", "ccqqc", "2.5", "0", "--n", "1", "--irange", TEN],
        cwd=work, env=env, logname="export_old_reference")
    export_state(work, "ccqqc", "1", TEN, "old_reference", dev)


def exports_old(dev):
    export_state(METHOD, "ccqq", "2", TEN, "old_reference", dev)
    export_state(METHOD, "bbqq", "1", TEN, "old_reference", dev)
    old_pcc_reference(dev)


def exports_redteam(dev):
    for name in ("reference_seed_0", "reference_seed_1", "reference_seed_2", "weaker_less_training",
                 "better_more_training"):
        irange = "[20, 21, 22, 23, 24, 25, 26, 27, 28, 29]" if name == "better_more_training" else TEN
        for system in ("ccqq", "bbqq"):
            export_state(RT_WORK / name / "DeepQuark", system, "1", irange, f"rt_{name}", dev)
    export_state(METHOD, "ccqqc", "1", "[25, 27, 43]", "old_pcc_continued", dev)


def evaluate(variant, dev, slices=("test", "train"), seed=None, root=None, tag=None):
    root = root or (ART / variant)
    tag = tag or variant
    for sl in slices:
        script = TEST_EVAL if sl == "test" else TRAIN_EVAL
        cmd = [sys.executable, script, "--artifacts", root, "--output", RES / f"{tag}_{sl}.json",
               "--chain-means", CHAINS / f"{tag}_{sl}.json"]
        if seed is not None:
            cmd += ["--seed", str(seed)]
        run(cmd, env=dev_env(dev), logname=f"eval_{tag}")


def crosscheck(system, run_id, irange, variant, dev):
    env = {**dev_env(dev), "CC_SUB": str(ART / variant / f"deepquark_{system}" / "wavefunction.jaxexport"),
           "CC_EVAL": str(TEST_EVAL), "CC_OUT": str(OUT / f"crosscheck_{system}.json")}
    run([sys.executable, ROOT / "scripts/hardening/crosscheck.py", "AL1", system, SPIN[system], "0",
         "--n", run_id, "--irange", irange], cwd=METHOD, env=env, logname=f"crosscheck_{system}")


SHORTCUTS = ["legacy_threshold_replay", "legacy_fabricated", "legacy_impossible", "corrupt", "empty", "bloated",
             "metadata_claims", "abs_sign", "noisy", "bosonic", "gaussian", "stencil"]


def build_shortcut(kind, dev):
    for system in ("ccqq", "bbqq", "ccqqc"):
        out = ART / f"sc_{kind}" / f"deepquark_{system}"
        if out.exists():
            shutil.rmtree(out)
        env = {**dev_env(dev), "SC_KIND": kind, "SC_SYSTEM": system, "SC_OUT": str(out),
               "SC_BASE": str(ART / "old_reference" / f"deepquark_{system}" / "wavefunction.jaxexport")}
        if kind == "bosonic":
            run_id, irange = {"ccqq": ("2", TEN), "bbqq": ("1", TEN), "ccqqc": ("1", "[25, 27, 43]")}[system]
            cmd = [sys.executable, ROOT / "scripts/hardening/shortcuts.py", "AL1", system, SPIN[system], "0",
                   "--n", run_id, "--irange", irange]
            run(cmd, cwd=METHOD, env=env, logname=f"sc_{kind}")
        else:
            run([sys.executable, ROOT / "scripts/hardening/shortcuts.py"], env=env, logname=f"sc_{kind}")


def shortcut_stage(kind, dev):
    build_shortcut(kind, dev)
    if kind == "metadata_claims":
        # forged score file at the output location must be overwritten
        forged = {"slice": "test", "experiments": {e: {"results": {"deepquark": {"binding_mev": -1e9}}}
                                                   for e in ("tcc_al1_binding", "tbb_al1_binding",
                                                             "pcc_al1_binding")}}
        (RES / f"sc_{kind}_test.json").write_text(json.dumps(forged))
    evaluate(f"sc_{kind}", dev, slices=("test",))


def official_scoring(dev):
    env = dev_env(dev)
    run(["bash", "scripts/evaluate_train.sh"], env=env, logname="official")
    run(["bash", "scripts/evaluate_test.sh"], env=env, logname="official")
    for sl in ("train", "test"):
        shutil.copy2(ROOT / f"scoring/scores_{sl}.json", RES / f"reference_{sl}.json")
        shutil.copy2(ROOT / f"scoring/scores_{sl}_details.json", RES / f"reference_{sl}_details.json")
    run([sys.executable, TRAIN_EVAL, "--artifacts", "checkpoints", "--output", "scoring/scores.json"],
        env=env, logname="official")


def plan(dev):
    """(name, function, estimated minutes, required stages) in priority order."""
    P = []

    def add(name, fn, minutes, requires=()):
        P.append((name, fn, minutes, tuple(requires)))

    # 1. evaluator validation on the previously shipped reference (also the 'before' numbers)
    add("exports_old", lambda: exports_old(dev), 20)
    add("eval_old_reference", lambda: evaluate("old_reference", dev), 25, ["exports_old"])
    add("crosscheck_ccqq", lambda: crosscheck("ccqq", "2", TEN, "old_reference", dev), 8, ["exports_old"])
    # 2. the reference every future attempt is ranked against
    for system, minutes in (("ccqqc", 55), ("ccqq", 55), ("bbqq", 45)):
        add(f"ref_{system}", lambda s=system: ref_system(s, dev), minutes)
    add("official_scoring", lambda: official_scoring(dev), 30, ["ref_ccqqc", "ref_ccqq", "ref_bbqq"])
    # 3. weaker / longer-trained / seed variants trained by the red-team audit
    add("exports_redteam", lambda: exports_redteam(dev), 12)
    add("crosscheck_ccqqc", lambda: crosscheck("ccqqc", "1", "[25, 27, 43]", "old_pcc_continued", dev), 8,
        ["exports_redteam"])
    for variant in ("rt_weaker_less_training", "rt_better_more_training", "old_pcc_continued",
                    "rt_reference_seed_0", "rt_reference_seed_1", "rt_reference_seed_2"):
        add(f"eval_{variant}", lambda v=variant: evaluate(v, dev, slices=("test",)), 8, ["exports_redteam"])
    # 4. training-seed repeats and the better variant (full stage-2 schedule, 5x final batch)
    add("seed1_ccqqc", lambda: train_variant("ref_seed1", "ccqqc", dict(REF_ENV, DQ_SEED="1"), dev), 55)
    add("better_ccqqc", lambda: train_variant("better_full_schedule", "ccqqc", BETTER_ENV, dev), 130)
    add("seed1_ccqq", lambda: train_variant("ref_seed1", "ccqq", dict(REF_ENV, DQ_SEED="1"), dev), 55)
    add("better_ccqq", lambda: train_variant("better_full_schedule", "ccqq", BETTER_ENV, dev), 120)
    add("eval_better_full_schedule", lambda: evaluate("better_full_schedule", dev), 25,
        ["better_ccqqc", "better_ccqq"])
    add("eval_ref_seed1", lambda: evaluate("ref_seed1", dev, slices=("test",)), 15, ["seed1_ccqqc", "seed1_ccqq"])
    # 5. shortcuts against the new evaluator
    for kind in SHORTCUTS:
        add(f"shortcut_{kind}", lambda k=kind: shortcut_stage(k, dev), 6, ["exports_old"])
    # 6. evaluator-seed noise (paired re-measurements of the same submissions)
    for seed in (101, 202):
        add(f"noise_reference_{seed}", lambda s=seed: evaluate(
            "reference", dev, slices=("test",), seed=s, root=ROOT / "checkpoints", tag=f"reference_seed{s}"),
            12, ["official_scoring"])
        add(f"noise_better_{seed}", lambda s=seed: evaluate(
            "better_full_schedule", dev, slices=("test",), seed=s, tag=f"better_full_schedule_seed{s}"),
            10, ["eval_better_full_schedule"])
        add(f"noise_old_reference_{seed}", lambda s=seed: evaluate(
            "old_reference", dev, slices=("test",), seed=s, tag=f"old_reference_seed{s}"), 12, ["exports_old"])
    # 7. optional
    add("better_bbqq", lambda: train_variant("better_full_schedule", "bbqq", BETTER_ENV, dev), 110)
    add("eval_better_full_schedule_bbqq", lambda: evaluate("better_full_schedule", dev), 25, ["better_bbqq"])
    return P


def main():
    log(f"GPUs visible: {gpu_count()}; budget {BUDGET / 3600:.2f} h")
    for system in ("ccqqc", "ccqq", "bbqq"):
        base_run(METHOD, system)
    for name, fn, minutes, requires in plan(0):
        if (STATE / f"{name}.done").exists():
            continue
        missing = [r for r in requires if not (STATE / f"{r}.done").exists()]
        if missing:
            log(f"skip {name}: requires {missing}")
            continue
        if time.time() - START + minutes * 60 > BUDGET:
            log(f"skip {name}: ~{minutes} min does not fit in the remaining time")
            continue
        stage(name, fn)
    summary = {p.stem: json.loads(p.read_text() or "{}") for p in sorted(STATE.glob("*.done"))}
    failed = {p.stem: p.read_text()[:500] for p in sorted(STATE.glob("*.failed"))}
    (OUT / "job_state.json").write_text(json.dumps({"done": summary, "failed": failed}, indent=2))
    print(json.dumps({"done": list(summary), "failed": failed}, indent=2))


if __name__ == "__main__":
    main()
