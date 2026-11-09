#!/usr/bin/env python3
"""Build shortcut / baseline submissions for the hardening audit.

Environment:
  SC_KIND   one of: legacy_fabricated, legacy_impossible, corrupt, empty,
            metadata_claims, abs_sign, noisy, bloated, bosonic, gaussian
  SC_SYSTEM ccqq | bbqq | ccqqc
  SC_BASE   path to a reference wavefunction.jaxexport (wrapper kinds)
  SC_OUT    submission directory to create

For SC_KIND=bosonic the DeepQuark model is rebuilt from method/ code, so the
script must then be started from method/DeepQuark with the usual positional
arguments of multiquark/eval.py (AL1 <quarks> <S> <I> --n RUN --irange ...).
"""
import json
import os
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import export as jexport  # noqa: E402

KIND = os.environ["SC_KIND"]
SYSTEM = os.environ["SC_SYSTEM"]
OUT = Path(os.environ["SC_OUT"])
OUT.mkdir(parents=True, exist_ok=True)
PLATFORMS = ("cpu", "cuda")
THRESH_GEV = {"ccqq": 3.878, "bbqq": 10.644, "ccqqc": 5.718}[SYSTEM]
N_PART = {"ccqq": 4, "bbqq": 4, "ccqqc": 5}[SYSTEM]
N_DISC = {"ccqq": 5, "bbqq": 5, "ccqqc": 4}[SYSTEM]
DIM = 3 * N_PART + N_DISC
WF = OUT / "wavefunction.jaxexport"


def write_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2) + "\n")


def legacy(emean, esigma, para):
    rec = {"Emean": emean, "Esigma": esigma, "color": "3*3:0.5,6*6:0.5"}
    write_json("results.json", {"0": {"1": rec, "2": rec}})
    write_json("parameters.json", {"hyperparams": {"para_num": para}})


def load_base():
    base = jexport.deserialize(bytearray(Path(os.environ["SC_BASE"]).read_bytes()))
    batch = base.in_avals[0].shape[0]
    return base, (batch if isinstance(batch, int) else None)


def export(fn, batch):
    if batch is None:
        (b,) = jexport.symbolic_shape("b")
        spec = jax.ShapeDtypeStruct((b, DIM), jnp.float64)
    else:
        spec = jax.ShapeDtypeStruct((batch, DIM), jnp.float64)
    exp = jexport.export(jax.jit(fn), platforms=PLATFORMS)(spec)
    WF.write_bytes(exp.serialize())


if KIND == "legacy_fabricated":
    legacy(THRESH_GEV - 1.0, 0.0, 1857)
elif KIND == "legacy_impossible":
    legacy(-1e6, -1e6, -1e9)
elif KIND == "corrupt":
    WF.write_bytes(np.random.default_rng(0).bytes(4096))
elif KIND == "empty":
    WF.write_bytes(b"")
elif KIND == "metadata_claims":
    WF.write_bytes(Path(os.environ["SC_BASE"]).read_bytes())
    legacy(-1e6, 0.0, 0)
    write_json("export_meta.json", {"threshold_mev": 1e9, "hamiltonian_identity_shift_gev": -0.1,
                                    "energy_mev": -1e9, "mc_error_mev": 0.0, "parameter_count": 0})
elif KIND == "abs_sign":
    # drop the fermionic sign: nodeless |psi|
    base, batch = load_base()
    export(lambda x: jnp.real(base.call(x)), batch)
elif KIND == "noisy":
    # deterministic but call-layout dependent perturbation of log psi
    base, _ = load_base()

    def noisy(x):
        noise = jax.random.normal(jax.random.PRNGKey(7), (x.shape[0],))
        return base.call(x) + 0.01 * noise

    export(noisy, 4096)
elif KIND == "bloated":
    base, batch = load_base()
    table = jnp.asarray(np.random.default_rng(1).normal(size=20000))

    def bloated(x):
        return base.call(x) + 0.0 * jnp.sum(table)

    export(bloated, batch)
elif KIND == "gaussian":
    # honest, untrained single-channel Gaussian in the correct sector
    n = N_PART
    allowed = {"ccqq": [0, 1, 0, 1, 0], "bbqq": [0, 1, 0, 1, 0], "ccqqc": [0, 1, 0, 1]}[SYSTEM]
    allowed = jnp.asarray(allowed, dtype=jnp.float64)
    width = 2.5  # GeV^-1

    def gaussian(x):
        R = x[:, :3 * n].reshape(x.shape[0], n, 3)
        d = R[:, :, None, :] - R[:, None, :, :]
        r2 = jnp.sum(d * d, axis=(1, 2, 3)) / 2.0
        ok = jnp.all(x[:, 3 * n:] == allowed[None, :], axis=-1)
        return jnp.where(ok, -r2 / (2.0 * width ** 2), -jnp.inf).astype(jnp.complex128)

    export(gaussian, None)
elif KIND == "bosonic":
    # rebuild the trained DeepQuark net with the exchange signs of bosons
    here = Path.cwd()
    sys.path.insert(0, str(here / "multiquark"))
    sys.path.insert(0, str(here))
    import export as dq_export  # noqa: E402  (method/DeepQuark/multiquark/export.py)

    logpath = dq_export.run_dir()
    index, _ = dq_export.select_checkpoint(logpath)
    good = dq_export.build_model()
    variables = dq_export.load_variables(good, logpath, index)
    model = dq_export.build_model(signs_override=tuple(1 for _ in good.signs))
    dq_export.export_function(lambda x: model.apply(variables, x), str(WF))
else:
    raise ValueError(KIND)

print(json.dumps({"kind": KIND, "system": SYSTEM, "out": str(OUT),
                  "files": sorted(p.name for p in OUT.iterdir())}))
