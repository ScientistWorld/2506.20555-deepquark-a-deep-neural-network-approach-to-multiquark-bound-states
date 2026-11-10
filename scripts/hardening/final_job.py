#!/usr/bin/env python3
"""Final hardening job: score the reference, its variants and the shortcuts with the
final evaluators (4x larger samples than the previous audit job), and train one
"better" variant (the paper's larger final-stage sample) under a time box.

Stages run one at a time in priority order; a stage starts only if its estimate fits
in the remaining time, and training/evaluation subprocesses of the variant stages are
killed at their time box (completed 50-iteration blocks are kept).  Finished stages
leave a marker in tmp/hardening/state_final, so resubmitting continues the plan.
Results: notes/hardening/final/*.json; logs: tmp/hardening/logs_final/.
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
STATE = HW / "state_final"
LOGS = HW / "logs_final"
CHAINS = HW / "chains_final"
OUT = ROOT / "notes/hardening/final"
METHOD = ROOT / "method/DeepQuark"
RT_WORK = ROOT / "redteam/work"
TEST_EVAL = ROOT / "eval/test/evaluate.py"
TRAIN_EVAL = ROOT / "eval/train/evaluate.py"
SHORTCUTS = ROOT / "scripts/hardening/shortcuts.py"
START = time.time()
BUDGET = float(os.environ.get("HW_BUDGET_HOURS", "5.6")) * 3600

SPIN = {"ccqq": "1", "bbqq": "1", "ccqqc": "2.5"}
PARENT = {"ccqq": "AL1/ccqq(S=1.0_I=0.0)", "bbqq": "AL1/bbqq(S=1.0_I=0.0)", "ccqqc": "AL1/ccqqc(S=2.5_I=0.0)"}
SYSTEMS = ("ccqq", "bbqq", "ccqqc")

# reference states (installed in checkpoints/): run directory and selected final-stage checkpoint
REF_RUN = {"ccqq": (METHOD / PARENT["ccqq"] / "3", 6),      # DQ_SEED=0 run (= redteam reference_seed_0)
           "bbqq": (METHOD / PARENT["bbqq"] / "2", 7),      # DQ_SEED=0 run (= redteam reference_seed_0)
           "ccqqc": (METHOD / PARENT["ccqqc"] / "1", 27)}   # base schedule + lr-1e-4 refinement

# better variant: the paper's "increase the sample size as the wave function approaches
# convergence" -- 10 more final-stage blocks of 50 SR iterations with 1e5 instead of 2e4
# samples, warm-started from the reference state, then the method's own (larger) evaluation
# selects among the start and the new checkpoints.
BETTER_LR = {"ccqq": "0.001", "bbqq": "0.001", "ccqqc": "0.0001"}
BETTER_NEW0 = 100
BETTER_BLOCKS = 10
BETTER_TRAIN_BOX = float(os.environ.get("HW_TRAIN_BOX_MIN", "30")) * 60
BETTER_EVAL_BOX = float(os.environ.get("HW_EVAL_BOX_MIN", "18")) * 60

for d in (ART, WORK, STATE, LOGS, CHAINS, OUT):
    d.mkdir(parents=True, exist_ok=True)

GPU_ENV = {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - START) / 60:6.1f}m] {msg}", flush=True)


def remaining():
    return BUDGET - (time.time() - START)


def pick_gpu():
    """With several GPUs visible and no CUDA_VISIBLE_DEVICES, use the least loaded one."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=60).stdout
    except Exception as exc:  # noqa: BLE001
        log(f"nvidia-smi failed: {exc!r}")
        return
    log("GPUs:\n" + out.strip())
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        log(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} set by the scheduler; using it")
        return
    rows = []
    for line in out.strip().splitlines():
        try:
            idx, _, mem, util = [p.strip() for p in line.split(",")]
            rows.append((float(util), float(mem), int(idx)))
        except ValueError:
            continue
    if len(rows) > 1:
        util, mem, idx = min(rows)
        GPU_ENV["JAX_CUDA_VISIBLE_DEVICES"] = str(idx)
        log(f"{len(rows)} GPUs visible; using GPU {idx} (util {util}%, {mem} MiB used)")


def run(cmd, cwd=ROOT, env=None, logname="misc", timeout=None, check=True):
    full = os.environ.copy()
    full.update(GPU_ENV)
    full.update(env or {})
    log(f"RUN {' '.join(map(str, cmd))}  (cwd={cwd}{', timeout %.0f s' % timeout if timeout else ''})")
    with open(LOGS / f"{logname}.log", "a") as handle:
        handle.write(f"\n===== {time.ctime()} {' '.join(map(str, cmd))}\n")
        handle.flush()
        try:
            proc = subprocess.run(list(map(str, cmd)), cwd=cwd, env=full, stdout=handle,
                                  stderr=subprocess.STDOUT, timeout=timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            log(f"time box reached for {logname}")
            return "timeout"
    if rc != 0 and check:
        tail = (LOGS / f"{logname}.log").read_text()[-3000:]
        raise RuntimeError(f"rc={rc} for {cmd}\n{tail}")
    return rc


def stage(name, fn, minutes):
    marker = STATE / f"{name}.done"
    if marker.exists():
        log(f"skip {name} (done)")
        return
    if remaining() < minutes * 60:
        log(f"skip {name}: ~{minutes} min does not fit ({remaining() / 60:.0f} min left)")
        (STATE / f"{name}.skipped").write_text(f"needed {minutes} min, had {remaining() / 60:.0f}")
        return
    log(f"start {name} (~{minutes} min)")
    t = time.time()
    try:
        info = fn() or {}
    except Exception as exc:  # noqa: BLE001
        log(f"FAILED {name}: {exc!r}"[:4000])
        (STATE / f"{name}.failed").write_text(repr(exc)[:20000])
        return
    marker.write_text(json.dumps({"seconds": time.time() - t, **info}))
    (STATE / f"{name}.failed").unlink(missing_ok=True)
    log(f"done {name} in {time.time() - t:.0f}s")


# ------------------------------------------------------------------ evaluation
def evaluate(root, tag, slices=("test",), seed=None):
    for sl in slices:
        script = TEST_EVAL if sl == "test" else TRAIN_EVAL
        cmd = [sys.executable, script, "--artifacts", root, "--output", OUT / f"{tag}_{sl}.json",
               "--chain-means", CHAINS / f"{tag}_{sl}.json"]
        if seed is not None:
            cmd += ["--seed", str(seed)]
        run(cmd, logname=f"eval_{tag}")


def official_scoring():
    run(["bash", "scripts/evaluate_test.sh"], logname="official")
    run(["bash", "scripts/evaluate_train.sh"], logname="official")
    for sl in ("train", "test"):
        shutil.copy2(ROOT / f"scoring/scores_{sl}.json", OUT / f"reference_{sl}.json")
        shutil.copy2(ROOT / f"scoring/scores_{sl}_details.json", OUT / f"reference_{sl}_details.json")
    # scripts/evaluate.sh writes scores.json with the public (train) protocol: same numbers
    shutil.copy2(ROOT / "scoring/scores_train.json", ROOT / "scoring/scores.json")
    shutil.copy2(ROOT / "scoring/scores_train_details.json", ROOT / "scoring/scores_details.json")


# ------------------------------------------------------------------ better variant
def better_system(system):
    """Warm-started larger-sample final stage for one system (time boxed)."""
    work = WORK / "better_large_batch" / "DeepQuark"
    if not work.exists():
        shutil.copytree(METHOD, work, ignore=shutil.ignore_patterns("AL1", "__pycache__"))
    src_run, ck = REF_RUN[system]
    dst = work / PARENT[system] / "1"
    if not dst.exists():
        dst.mkdir(parents=True)
        shutil.copy2(src_run / f"Train3_{ck}.mpack", dst / f"Train3_{ck}.mpack")
        for f in ("train.json", "parameters.json"):
            shutil.copy2(src_run / f, dst / f)
    args = ["AL1", system, SPIN[system], "0", "--n", "1"]
    new = sorted(i for i in range(BETTER_NEW0, BETTER_NEW0 + BETTER_BLOCKS) if (dst / f"Train3_{i}.mpack").exists())
    todo = BETTER_BLOCKS - len(new)
    info = {}
    if todo > 0:
        start = max(new) if new else ck
        first = (max(new) + 1) if new else BETTER_NEW0
        env = {"JAX_PLATFORM_NAME": "gpu", "DQ_SEED": "0", "DQ_NCHAINS": "2000", "DQ_NSWEEP": "20",
               "DQ_NSAMPLE": "100000", "DQ_TRAIN2_NDISCARD": "200", "DQ_RESUME_FROM": str(start),
               "DQ_RESUME_START_INDEX": str(first), "DQ_RESUME_TRAIN3_BLOCKS": str(todo),
               "DQ_RESUME_TRAIN3_ITERS": "50", "DQ_RESUME_LR": BETTER_LR[system],
               "DQ_RESUME_WARMUP_ITERS": "50" if not new else "0"}
        box = min(BETTER_TRAIN_BOX, max(60.0, remaining() - 25 * 60))
        t = time.time()
        rc = run([sys.executable, "multiquark/continue_train3.py", *args], cwd=work, env=env,
                 logname=f"better_{system}", timeout=box, check=False)
        info["train_seconds"] = time.time() - t
        info["train_rc"] = rc
    new = sorted(i for i in range(BETTER_NEW0, BETTER_NEW0 + BETTER_BLOCKS) if (dst / f"Train3_{i}.mpack").exists())
    info["new_checkpoints"] = new
    if not new:
        raise RuntimeError(f"no refined checkpoint for {system}")
    irange = str([ck] + new)
    env = {"JAX_PLATFORM_NAME": "gpu", "DQ_SEED": "0", "DQ_EVAL_NCHAINS": "2000", "DQ_EVAL_NSWEEP": "100",
           "DQ_EVAL_NSAMPLE": "400000", "DQ_EVAL_NDISCARD": "1000", "DQ_EVAL_REPEATS": "2",
           "DQ_EVAL_SKIP_RMS": "1", "DQ_EVAL_CHUNK": "25000"}
    rc = run([sys.executable, "multiquark/eval.py", *args, "--irange", irange], cwd=work, env=env,
             logname=f"better_{system}", timeout=min(BETTER_EVAL_BOX, max(60.0, remaining() - 15 * 60)),
             check=False)
    if rc != 0:
        raise RuntimeError(f"own evaluation of the refined {system} checkpoints did not finish (rc={rc})")
    out = ART / "better_large_batch" / f"deepquark_{system}"
    if out.exists():
        shutil.rmtree(out)
    run([sys.executable, METHOD / "multiquark/export.py", *args, "--irange", irange], cwd=work,
        env={"DQ_EXPORT_DIR": str(out)}, logname=f"better_{system}")
    info["export"] = json.loads((out / "export_meta.json").read_text())
    return info


# ------------------------------------------------------------------ shortcuts
def build_shortcut(kind):
    for system in SYSTEMS:
        out = ART / f"scf_{kind}" / f"deepquark_{system}"
        if out.exists():
            shutil.rmtree(out)
        env = {"SC_KIND": kind, "SC_SYSTEM": system, "SC_OUT": str(out),
               "SC_BASE": str(ROOT / "checkpoints" / f"deepquark_{system}" / "wavefunction.jaxexport")}
        if kind == "bosonic":
            run_dir, ck = REF_RUN[system]
            run([sys.executable, SHORTCUTS, "AL1", system, SPIN[system], "0", "--n", run_dir.name,
                 "--irange", str([ck])], cwd=METHOD, env=env, logname=f"scf_{kind}")
        else:
            run([sys.executable, SHORTCUTS], env=env, logname=f"scf_{kind}")


def shortcut_stage(kind):
    build_shortcut(kind)
    evaluate(ART / f"scf_{kind}", f"sc_{kind}")


# ------------------------------------------------------------------ seeded Pcc reproduction
def repro_pcc():
    work = WORK / "repro_pcc" / "DeepQuark"
    if not work.exists():
        shutil.copytree(METHOD, work, ignore=shutil.ignore_patterns("AL1", "__pycache__"))
    env = {"DQ_SYSTEMS": "ccqqc", "DQ_SKIP_EVALUATE": "1", "DQ_CODE_DIR": str(work),
           "DQ_ARTIFACT_ROOT": str(ART / "repro_pcc")}
    rc = run(["bash", "scripts/reproduce.sh"], env=env, logname="repro_pcc",
             timeout=max(60.0, remaining() - 12 * 60), check=False)
    if rc != 0:
        raise RuntimeError(f"seeded Pcc reproduction did not finish (rc={rc})")
    evaluate(ART / "repro_pcc", "repro_pcc")


def plan():
    P = []

    def add(name, fn, minutes):
        P.append((name, fn, minutes))

    # 1. the reference, scored by the final evaluators (writes scoring/scores_{train,test}.json)
    add("official_scoring", official_scoring, 15)
    # 2. before/after and weaker variants on the final hidden test
    add("eval_shipped_reference", lambda: evaluate(ART / "old_reference", "shipped_reference"), 8)
    add("eval_weaker_less_training", lambda: evaluate(ART / "rt_weaker_less_training", "weaker_less_training"), 6)
    # 3. redteam shortcuts (artifact-free JSON claims) and malformed submissions
    for kind in ("legacy_threshold_replay", "legacy_fabricated", "legacy_impossible", "corrupt", "empty",
                 "bloated"):
        add(f"shortcut_{kind}", lambda k=kind: shortcut_stage(k), 3)
    # 4. better variant: larger final-stage sample, warm-started from the reference states
    for system in ("ccqqc", "ccqq", "bbqq"):
        add(f"better_{system}", lambda s=system: better_system(s), 20)
    add("eval_better_large_batch",
        lambda: evaluate(ART / "better_large_batch", "better_large_batch", slices=("test", "train")), 10)
    # 5. noise: evaluator seed (paired re-measurements) and training seed
    for seed in (101, 202):
        add(f"noise_reference_{seed}",
            lambda s=seed: evaluate(ROOT / "checkpoints", f"reference_seed{s}", seed=s), 8)
    for name in ("rt_reference_seed_1", "rt_reference_seed_2"):
        add(f"eval_{name}", lambda n=name: evaluate(ART / n, n), 6)
    add("eval_rt_better_more_training", lambda: evaluate(ART / "rt_better_more_training", "rt_better_more_training"), 6)
    add("noise_better_large_batch_101",
        lambda: evaluate(ART / "better_large_batch", "better_large_batch_seed101", seed=101), 8)
    # 6. wavefunction-level shortcuts against the final evaluator
    for kind in ("metadata_claims", "kink", "abs_sign", "bosonic", "noisy", "stencil", "gaussian"):
        add(f"shortcut_{kind}", lambda k=kind: shortcut_stage(k), 8)
    # 7. seeded from-scratch Pcc reproduction with scripts/reproduce.sh (fast nodes only)
    add("repro_pcc", repro_pcc, 60)
    return P


def main():
    pick_gpu()
    log(f"budget {BUDGET / 3600:.2f} h")
    for name, fn, minutes in plan():
        stage(name, fn, minutes)
    summary = {p.stem: json.loads(p.read_text() or "{}") for p in sorted(STATE.glob("*.done"))}
    failed = {p.stem: p.read_text()[:1500] for p in sorted(STATE.glob("*.failed"))}
    skipped = {p.stem: p.read_text() for p in sorted(STATE.glob("*.skipped"))
               if not (STATE / f"{p.stem}.done").exists()}
    state = {"done": summary, "failed": failed, "skipped": skipped,
             "hours": (time.time() - START) / 3600, "gpu_env": GPU_ENV}
    (OUT / "job_state.json").write_text(json.dumps(state, indent=2, default=str))
    print(json.dumps(state, indent=2, default=str))


if __name__ == "__main__":
    main()
