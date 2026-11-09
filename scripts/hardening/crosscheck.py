#!/usr/bin/env python3
"""Cross-check the gym evaluator's local energies against NetKet's.

Run from method/DeepQuark with the eval.py positional arguments, e.g.

  CC_SUB=.../wavefunction.jaxexport CC_EVAL=/home/user/eval/test/evaluate.py \
  CC_OUT=notes/hardening/crosscheck_ccqq.json \
  python /home/user/scripts/hardening/crosscheck.py AL1 ccqq 1 0 --n 2 --irange "[0,...,9]"

The same configurations are scored with (a) the evaluator's finite-difference
local energy of the exported wavefunction and (b) NetKet's autodiff local
energy of the DeepQuark model with the AL1 Hamiltonian of the method code.
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

here = Path.cwd()
sys.path.insert(0, str(here / "multiquark"))
sys.path.insert(0, str(here))

import export as dq_export  # noqa: E402  (also parses the positional args)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from hi_op_ import ha, mulpot, kin  # noqa: E402
from config_ import quarks  # noqa: E402

spec = importlib.util.spec_from_file_location("gym_eval", os.environ["CC_EVAL"])
gym = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gym)

name = {"ccqq": "tcc_al1_binding", "bbqq": "tbb_al1_binding", "ccqqc": "pcc_al1_binding"}[quarks]
cfg = gym.build_system(name)
t0 = time.time()

# configurations from the evaluator's own sampler (short run)
gym.SAMPLES_PER_CHAIN = 4
gym.ADAPT_BLOCKS = 20
gym.EXTRA_BURN_STEPS = 500
sub = gym.load_submission(Path(os.environ["CC_SUB"]), cfg)
call = gym.make_logpsi(sub, cfg)
rows_raw, rows_eval = gym.make_rows_eval(call, cfg["dim"])
sample = gym.make_sampler(call, cfg)
build_rows, energy_block, K = gym.make_energy_fns(cfg, rows_raw)
Rs, as_, stats = sample(jax.random.PRNGKey(4242))
n = cfg["n"]
R = np.asarray(Rs).reshape(-1, n, 3)
a = np.asarray(as_).reshape(-1)
N = R.shape[0]
G = N // gym.ENERGY_CHUNK
keys = jax.vmap(lambda c: jax.random.fold_in(jax.random.PRNGKey(99), c))(jnp.arange(G))
e_gym, t_gym = energy_block(jnp.asarray(R.reshape(G, gym.ENERGY_CHUNK, n, 3)),
                            jnp.asarray(a.reshape(G, gym.ENERGY_CHUNK)), keys)
e_gym = np.asarray(e_gym).reshape(-1)
t_gym = np.asarray(t_gym).reshape(-1)

# NetKet local energies of the same configurations with the method's model
logpath = dq_export.run_dir()
index, own_energy = dq_export.select_checkpoint(logpath)
model = dq_export.build_model()
variables = dq_export.load_variables(model, logpath, index)
x = np.concatenate([R.reshape(N, -1), cfg["onehot"][a]], axis=1)

# the exported function must equal the model
lp_model = np.asarray(jax.jit(lambda v: model.apply(variables, v))(jnp.asarray(x[:4096])))
lp_export = np.asarray(rows_eval(jnp.asarray(x[:4096])))
export_diff = float(np.max(np.abs(lp_model - lp_export)))


def nk_local(op, xs):
    """NetKet local estimator of op on fixed configurations xs (N, D)."""
    import netket as nk
    from sampler_ import GaussianFlipRule
    from hi_op_ import hi, hi_dis
    from config_ import sigma

    errors = []
    try:
        n_chains = 16
        sa = nk.sampler.MetropolisSampler(hi, GaussianFlipRule(hi_dis=hi_dis, sigma=sigma), n_chains=n_chains)
        vs = nk.vqs.MCState(sa, model, n_samples=xs.shape[0], n_discard_per_chain=0, chunk_size=2048)
        vs.variables = variables
        vs._samples = jnp.asarray(xs.reshape(n_chains, -1, xs.shape[-1]))
        return np.asarray(vs.local_estimators(op)).reshape(-1).real
    except Exception as exc:  # noqa: BLE001
        errors.append(repr(exc))
    args = op._pack_arguments()
    patterns = [
        jax.jit(lambda c: op._expect_kernel(model.apply, variables, c, args)),
        jax.jit(jax.vmap(lambda xi: op._expect_kernel(model.apply, variables, xi, args))),
        jax.jit(jax.vmap(lambda xi: jnp.reshape(op._expect_kernel(model.apply, variables, xi[None], args), (-1,))[0])),
    ]
    for fn in patterns:
        try:
            out = [np.asarray(fn(jnp.asarray(xs[i:i + 2048]))).reshape(-1) for i in range(0, xs.shape[0], 2048)]
            return np.concatenate(out).real
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc)[:500])
    raise RuntimeError(" | ".join(errors))


report = {"system": quarks, "checkpoint": int(index), "n_samples": int(N), "sampler": stats,
          "export_vs_model_max_abs_logpsi_diff": export_diff}
try:
    e_pot = nk_local(mulpot, x)
    e_kin = nk_local(kin, x)
    e_nk = e_pot + e_kin
    diff = e_gym - e_nk
    kdiff = t_gym - e_kin
    report.update({
        "netket_mean_mev": float(e_nk.mean() * 1000), "gym_mean_mev": float(e_gym.mean() * 1000),
        "mean_diff_mev": float(diff.mean() * 1000),
        "median_abs_diff_mev": float(np.median(np.abs(diff)) * 1000),
        "p99_abs_diff_mev": float(np.quantile(np.abs(diff), 0.99) * 1000),
        "kinetic_mean_diff_mev": float(kdiff.mean() * 1000),
        "kinetic_median_abs_diff_mev": float(np.median(np.abs(kdiff)) * 1000),
        "local_energy_std_mev": float(e_nk.std() * 1000),
    })
except Exception as exc:  # noqa: BLE001
    report["netket_error"] = repr(exc)
report["seconds"] = time.time() - t0
Path(os.environ["CC_OUT"]).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
