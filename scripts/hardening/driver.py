#!/usr/bin/env python3
"""Hardening job: retrain the reference, train variants, build shortcuts, score all.

Three tracks run concurrently (each stage is resumable through marker files):
  ref     reference DeepQuark (scripts/reproduce.sh, seed 0) for ccqqc, ccqq, bbqq,
          then reference seed 1 for ccqqc and ccqq (training-seed noise)
  better  full-schedule variant (no early stop in stage 2, 5x final-stage batch)
  eval    exports of previously trained states, NetKet cross-check, shortcut
          submissions, evaluator-seed noise, and finally the official scoring.

Usage: python scripts/hardening/driver.py            (all tracks)
       python scripts/hardening/driver.py <track>    (one track)
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
BUDGET = 5.6 * 3600

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
def train_variant(variant, system, env, dev):
    work = WORK / variant / "DeepQuark"
    if not work.exists():
        shutil.copytree(METHOD, work, ignore=shutil.ignore_patterns("AL1", "__pycache__"))
    full = dict(env)
    full.update(dev_env(dev))
    full.update({"JAX_PLATFORM_NAME": "gpu", "DQ_CODE_DIR": str(work), "DQ_POTENTIAL": "AL1",
                 "DQ_QUARKS": system, "DQ_SPIN": SPIN[system], "DQ_ISOSPIN": "0",
                 "DQ_ARTIFACT_NAME": f"deepquark_{system}", "DQ_ARTIFACT_ROOT": str(ART / variant)})
    run(["bash", "scripts/method.sh"], env=full, logname=f"{variant}_{system}")


def track_ref(dev):
    for system in ("ccqqc", "ccqq", "bbqq"):
        stage(f"ref_{system}", lambda s=system: run(
            ["bash", "scripts/reproduce.sh"],
            env={**dev_env(dev), "DQ_SYSTEMS": s, "DQ_SKIP_EVALUATE": "1"}, logname=f"ref_{s}"))
    for system in ("ccqqc", "ccqq"):
        stage(f"seed1_{system}", lambda s=system: train_variant("ref_seed1", s, dict(REF_ENV, DQ_SEED="1"), dev),
              optional_after=3.6 * 3600)


def track_better(dev):
    for system in ("ccqqc", "ccqq", "bbqq"):
        stage(f"better_{system}", lambda s=system: train_variant("better_full_schedule", s, BETTER_ENV, dev),
              optional_after=3.3 * 3600 if system == "bbqq" else 2.4 * 3600)


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


SHORTCUTS = ["legacy_fabricated", "legacy_impossible", "corrupt", "empty", "bloated", "metadata_claims",
             "abs_sign", "noisy", "bosonic", "gaussian"]


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


def track_eval(dev):
    stage("exports_old", lambda: exports_old(dev))
    stage("eval_old_reference", lambda: evaluate("old_reference", dev))
    stage("crosscheck_ccqq", lambda: crosscheck("ccqq", "2", TEN, "old_reference", dev))
    stage("exports_redteam", lambda: exports_redteam(dev))
    stage("crosscheck_ccqqc", lambda: crosscheck("ccqqc", "1", "[25, 27, 43]", "old_pcc_continued", dev))
    for variant in ("old_pcc_continued", "rt_reference_seed_0", "rt_reference_seed_1", "rt_reference_seed_2",
                    "rt_better_more_training", "rt_weaker_less_training"):
        stage(f"eval_{variant}", lambda v=variant: evaluate(v, dev))
    for kind in SHORTCUTS:
        stage(f"shortcut_{kind}", lambda k=kind: shortcut_stage(k, dev))
    for seed in (101, 202, 303):
        stage(f"noise_old_reference_{seed}", lambda s=seed: evaluate(
            "old_reference", dev, slices=("test",), seed=s, tag=f"old_reference_seed{s}"))

    # official scoring of the freshly trained reference
    wait_for([f"ref_{s}" for s in ("ccqqc", "ccqq", "bbqq")], limit=5.0 * 3600)
    stage("official_scoring", lambda: official_scoring(dev))
    for seed in (101, 202, 303):
        stage(f"noise_reference_{seed}", lambda s=seed: evaluate(
            "reference", dev, slices=("test",), seed=s, root=ROOT / "checkpoints", tag=f"reference_seed{s}"),
            optional_after=5.3 * 3600)

    wait_for(["better_ccqqc", "better_ccqq", "seed1_ccqqc", "seed1_ccqq"], limit=5.2 * 3600)
    for variant in ("better_full_schedule", "ref_seed1"):
        stage(f"eval_{variant}", lambda v=variant: evaluate(v, dev), optional_after=5.45 * 3600)
    for seed in (101, 202, 303):
        stage(f"noise_better_{seed}", lambda s=seed: evaluate(
            "better_full_schedule", dev, slices=("test",), seed=s, tag=f"better_full_schedule_seed{s}"),
            optional_after=5.45 * 3600)
    wait_for(["better_bbqq"], limit=5.5 * 3600)
    if (STATE / "better_bbqq.done").exists():
        (STATE / "eval_better_full_schedule.done").unlink(missing_ok=True)
        stage("eval_better_full_schedule", lambda: evaluate("better_full_schedule", dev),
              optional_after=5.6 * 3600)


def main():
    if len(sys.argv) > 1:
        track, dev = sys.argv[1], int(sys.argv[2])
        {"ref": track_ref, "better": track_better, "eval": track_eval}[track](dev)
        return
    n_gpu = gpu_count()
    log(f"GPUs visible: {n_gpu}")
    second = 1 if n_gpu >= 2 else 0
    env = {"HW_START": str(START)}
    procs = []
    for track, dev in (("ref", 0), ("better", second), ("eval", 0)):
        handle = open(LOGS / f"track_{track}.log", "a")
        procs.append((track, handle, subprocess.Popen(
            [sys.executable, __file__, track, str(dev)], stdout=handle, stderr=subprocess.STDOUT,
            env={**os.environ, **env})))
        time.sleep(5)
    while procs:
        for item in list(procs):
            track, handle, proc = item
            if proc.poll() is not None:
                log(f"track {track} finished rc={proc.returncode}")
                handle.close()
                procs.remove(item)
        time.sleep(30)
    for track in ("ref", "better", "eval"):
        print(f"===== track {track} =====")
        print((LOGS / f"track_{track}.log").read_text()[-6000:])
    summary = {p.stem: json.loads(p.read_text() or "{}") for p in sorted(STATE.glob("*.done"))}
    failed = {p.stem: p.read_text()[:500] for p in sorted(STATE.glob("*.failed"))}
    (OUT / "job_state.json").write_text(json.dumps({"done": summary, "failed": failed}, indent=2))
    print(json.dumps({"done": list(summary), "failed": failed}, indent=2))


if __name__ == "__main__":
    main()
