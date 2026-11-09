"""Export a trained DeepQuark wavefunction in the gym submission format.

Usage (same positional arguments as train.py / eval.py):

    python multiquark/export.py AL1 ccqq 1 0 --n RUN --irange "[0, 1, ..., 9]"

The checkpoint with the lowest energy in the method's own evaluation
(results.json written by eval.py) is selected, as in the paper's final stage,
and serialized with ``jax.export`` to ``$DQ_EXPORT_DIR/wavefunction.jaxexport``.
``DQ_EXPORT_INDEX`` forces a specific Train3 checkpoint.
"""
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flax  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import export as jexport  # noqa: E402

from model_ import QQqq_FCN, Q1Q2qq_FCN, exchange_indices, Penta_FCN  # noqa: E402
from config_ import quarks, MASS, S, I, out_path, nlayers, nhid, bound, n, irange  # noqa: E402
from hi_op_ import _nd, _nparticles, hi, n_discrete  # noqa: E402

FIXED_BATCH = 4096


def build_model(signs_override=None):
    if quarks[0] == quarks[1] and quarks[2] == quarks[3]:
        idparticles = [[0, 1, -1], [2, 3, (-1) ** I]]
    elif quarks[2] == quarks[3]:
        idparticles = [[2, 3, (-1) ** I]]
    else:
        raise ValueError("Not implemented case of identical particles")
    swap_indices, signs = exchange_indices(_nparticles, identical_particles=idparticles)
    if signs_override is not None:
        signs = tuple(signs_override)
    kwargs = dict(nd=_nd, nparticles=_nparticles, mass=tuple(MASS), ndiscrete=n_discrete, nhid=nhid,
                  nlayers=nlayers, bound=bound, indices=swap_indices, signs=signs)
    if _nparticles == 4:
        return QQqq_FCN(**kwargs) if quarks[0] == quarks[1] else Q1Q2qq_FCN(**kwargs)
    if _nparticles == 5:
        return Penta_FCN(**kwargs)
    raise ValueError(f"Unsupported particle count: {_nparticles}")


def run_dir():
    if n is None:
        raise ValueError("Pass --n with the run directory")
    return out_path('eval', n)


def select_checkpoint(logpath):
    forced = os.environ.get("DQ_EXPORT_INDEX")
    with open(f"{logpath}/results.json", "r", encoding="utf-8") as handle:
        results = json.load(handle)
    candidates = []
    for ck, repeats in results.items():
        if irange and int(ck) not in irange:
            continue
        for rec in repeats.values():
            candidates.append((float(rec["Emean"]), int(ck)))
    if forced is not None:
        own = [e for e, ck in candidates if ck == int(forced)]
        return int(forced), (min(own) if own else None)
    if not candidates:
        raise ValueError(f"No evaluated checkpoints in {logpath}/results.json")
    energy, ck = min(candidates)
    return ck, energy


def load_variables(model, logpath, index):
    template = model.init(jax.random.PRNGKey(0), hi.random_state(key=jax.random.PRNGKey(0), size=1))
    with open(f"{logpath}/Train3_{index}.mpack", "rb") as handle:
        return flax.serialization.from_bytes(template, handle.read())


def input_dim():
    return _nparticles * _nd + n_discrete


def export_function(logpsi, path, platforms=("cpu", "cuda")):
    """Serialize logpsi: float64[batch, dim] -> complex128/float64[batch]."""
    dim = input_dim()
    fn = jax.jit(logpsi)
    try:
        (b,) = jexport.symbolic_shape("b")
        exported = jexport.export(fn, platforms=platforms)(jax.ShapeDtypeStruct((b, dim), jnp.float64))
        batch = "symbolic"
    except Exception as exc:  # noqa: BLE001
        print(f"symbolic batch export not possible ({type(exc).__name__}); using fixed batch {FIXED_BATCH}",
              flush=True)
        exported = jexport.export(fn, platforms=platforms)(jax.ShapeDtypeStruct((FIXED_BATCH, dim), jnp.float64))
        batch = FIXED_BATCH
    blob = exported.serialize()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(blob)

    # round-trip check against the in-memory function
    restored = jexport.deserialize(bytearray(blob))
    x = np.asarray(hi.random_state(key=jax.random.PRNGKey(1), size=FIXED_BATCH), dtype=np.float64)
    direct = np.asarray(fn(jnp.asarray(x)))
    roundtrip = np.asarray(jax.jit(restored.call)(jnp.asarray(x)))
    finite = np.isfinite(direct) & np.isfinite(roundtrip)
    if not np.array_equal(np.isfinite(direct), np.isfinite(roundtrip)) or not finite.any():
        raise RuntimeError("exported function disagrees with the model on finiteness")
    err = float(np.max(np.abs(direct[finite] - roundtrip[finite])))
    if not err < 1e-10:
        raise RuntimeError(f"exported function disagrees with the model (max diff {err})")
    return {"batch": batch, "bytes": len(blob), "roundtrip_max_diff": err}


def main():
    out_dir = os.environ.get("DQ_EXPORT_DIR")
    if not out_dir:
        raise ValueError("Set DQ_EXPORT_DIR to the submission directory")
    logpath = run_dir()
    index, own_energy = select_checkpoint(logpath)
    model = build_model()
    variables = load_variables(model, logpath, index)

    def logpsi(x):
        return model.apply(variables, x)

    meta = export_function(logpsi, os.path.join(out_dir, "wavefunction.jaxexport"))
    with open(f"{logpath}/parameters.json", "r", encoding="utf-8") as handle:
        params = json.load(handle)
    meta.update({
        "system": quarks, "S": S, "I": I, "run": n, "train3_checkpoint": index,
        "method_eval_energy_gev": own_energy,
        "trainable_parameters": params.get("hyperparams", {}).get("para_num"),
    })
    with open(os.path.join(out_dir, "export_meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
