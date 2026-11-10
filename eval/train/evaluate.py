#!/usr/bin/env python3
"""Public train-slice scorer for the AL1 multiquark bound-state gym.

Same measurement as the held-out test scorer, with a public sampler seed and a
quarter of the samples, so methods can be developed against it.

A submission for system <sys> is the file

    checkpoints/<method>_<sys>/wavefunction.jaxexport

holding a serialized JAX function (``jax.export``) that maps a batch of
configurations to log(psi).  Nothing a method writes besides that function is
read.  This evaluator owns everything on the other side of the comparison: the
AL1 Hamiltonian, the physical sector (exchange symmetry and parity), the
two-hadron thresholds, the Metropolis sampler, its random seed, the number of
samples and the acceptance gates.  Every scored number is recomputed here from
fresh samples of the submitted wavefunction.

A submission that is missing, malformed, outside the physical sector,
non-deterministic, over the parameter budget, non-finite, unconverged or
physically implausible is scored as failed: it receives the worst value of
every scored metric (``binding_mev = 0``, ``mc_error_mev`` at its cap) and
``valid = 0``.  A valid submission that finds no bound state (energy at or
above the threshold) has failed the task and gets the same scored values.

The submitted function is never shown more than one row of the same sample in
one call: local energies are evaluated in a sample-transposed layout (each
call holds one stencil point for each of 4096 different samples), so a
function cannot recognise finite-difference stencils or discrete-channel
probes.  The values it returns there must agree with the values it returned
to the sampler and with a re-evaluation in the natural layout.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402
from jax import export as jexport  # noqa: E402


# --------------------------------------------------------------------------
# Fixed evaluation protocol (train slice)
# --------------------------------------------------------------------------
SLICE = "train"
SEED = 12345
N_CHAINS = 4096
SAMPLES_PER_CHAIN = 128          # 524,288 samples per system
THIN = 10                        # Metropolis steps between stored samples
ADAPT_BLOCKS = 40                # step-size adaptation blocks during burn-in
ADAPT_BLOCK_STEPS = 50
EXTRA_BURN_STEPS = 1000          # burn-in after adaptation, fixed step size
INITIAL_STEP = 0.5               # GeV^-1
INITIAL_SPREAD = 1.5             # GeV^-1
CALL_BATCH = 4096                # every call of a submission sees this many rows
ENERGY_CHUNK = CALL_BATCH        # samples per local-energy chunk (one row per sample per call)
ENERGY_GROUP = 8                 # chunks per compiled call
FD_H_MIN, FD_H_MAX = 0.5e-3, 1.5e-3   # finite-difference step range, GeV^-1
SYM_SAMPLES = 65536
SYM_TOL = 1.0e-6                 # max squared relative L2 symmetry violation
DET_TOL = 1.0e-12                # max |delta log psi| between repeated identical calls
LAYOUT_ABS_TOL = 1.0e-6          # GeV; max local-energy difference between call layouts
LAYOUT_REL_TOL = 1.0e-9
CONSIST_TOL = 1.0e-8             # max |log|psi|| difference, sampler vs energy pass
RHAT_MAX = 1.2
PARAM_BUDGET = 10000             # stored constants, counted in float64 equivalents (8 bytes)
MC_ERROR_CAP_MEV = 5.0           # worst scored Monte Carlo error (also given to failed / unbound runs)
FILE_NAME = "wavefunction.jaxexport"
SUBMISSION_TIMEOUT = 3600        # seconds per (system, method); each runs in its own process
MAX_FILE_BYTES = 64 * 1024 * 1024

# --------------------------------------------------------------------------
# AL1 Hamiltonian (Semay & Silvestre-Brac), GeV units, hbar = c = 1
# --------------------------------------------------------------------------
AL1 = {"kappa": 0.5069, "lam": 0.1653, "Lambda": 0.8321, "kappa_p": 1.8609, "A": 1.6553, "B": 0.2204}
QUARK_MASS = {"b": 5.227, "c": 1.836, "q": 0.315}
SQ2 = math.sqrt(2.0)

SYSTEMS = {
    "tcc_al1_binding": {
        "suffix": "ccqq", "particles": "ccqq", "kind": "tetra",
        "threshold_mev": 1862.0 + 2016.0, "floor_mev": -100.0,
    },
    "tbb_al1_binding": {
        "suffix": "bbqq", "particles": "bbqq", "kind": "tetra",
        "threshold_mev": 5294.0 + 5350.0, "floor_mev": -400.0,
    },
    "pcc_al1_binding": {
        "suffix": "ccqqc", "particles": "ccqqc", "kind": "penta",
        "threshold_mev": 2016.0 + 3702.0, "floor_mev": -100.0,
    },
}


def _tetra_tables():
    """Isoscalar S=1 QQ qbar qbar sector: color x spin coupled basis."""
    colors = ["33", "66"]
    color_onehot = {"33": [0, 1], "66": [1, 0]}
    color_keys = {"33": "color_3bar3", "66": "color_6bar6"}
    spins = ["01", "10", "11"]  # [QQ]^sa [qbar qbar]^sb
    spin_onehot = {"01": [0, 0, 1], "10": [0, 1, 0], "11": [1, 0, 0]}
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    cgroup = {(0, 1): 0, (2, 3): 0, (0, 2): 1, (1, 3): 1, (0, 3): 2, (1, 2): 2}
    sidx = {(0, 1): 0, (2, 3): 1, (0, 2): 2, (1, 3): 3, (0, 3): 4, (1, 2): 5}
    s = 1.0 / (2.0 * SQ2)
    spin_table = {
        ("01", "01"): [-0.75, 0.25, 0, 0, 0, 0],
        ("10", "10"): [0.25, -0.75, 0, 0, 0, 0],
        ("11", "11"): [0.25, 0.25, -0.25, -0.25, -0.25, -0.25],
        ("11", "10"): [0, 0, s, -s, -s, s],
        ("11", "01"): [0, 0, -s, s, -s, s],
        ("10", "01"): [0, 0, 0.25, 0.25, -0.25, -0.25],
    }

    def color_factor(c1, c2, pair):
        g = cgroup[pair]
        if c1 == c2 == "33":
            return [-8 / 3, -4 / 3, -4 / 3][g]
        if c1 == c2 == "66":
            return [4 / 3, -10 / 3, -10 / 3][g]
        return [0.0, -2 * SQ2, 2 * SQ2][g]

    def spin_factor(s1, s2, pair):
        key = (s1, s2) if (s1, s2) in spin_table else (s2, s1)
        return spin_table[key][sidx[pair]]

    # exchange characters: color/spin basis sign under (12) and (34)
    c_sign = {"33": -1.0, "66": 1.0}
    s_sign12 = {"01": -1.0, "10": 1.0, "11": 1.0}
    s_sign34 = {"01": 1.0, "10": -1.0, "11": 1.0}
    states = [(c, sp) for c in colors for sp in spins]
    onehot = [color_onehot[c] + spin_onehot[sp] for c, sp in states]
    sym = [
        # (name, permutation, invert coordinates, expected eigenvalue per state)
        ("exchange_QQ", [1, 0, 2, 3], False, [-c_sign[c] * s_sign12[sp] for c, sp in states]),
        ("exchange_light", [0, 1, 3, 2], False, [c_sign[c] * s_sign34[sp] for c, sp in states]),
        ("parity", [0, 1, 2, 3], True, [1.0 for _ in states]),
    ]
    return states, onehot, pairs, color_factor, spin_factor, sym, colors, color_keys


def _penta_tables():
    """Isoscalar S=5/2 QQ qq Qbar sector: color basis x single spin state."""
    colors = ["33", "36", "63"]
    color_onehot = {"33": [0, 0, 1], "36": [0, 1, 0], "63": [1, 0, 0]}
    color_keys = {"33": "color_3bar3", "36": "color_3bar6", "63": "color_6bar3"}
    pidx = {(0, 1): 0, (2, 3): 1, (0, 2): 2, (1, 3): 3, (0, 3): 4, (1, 2): 5,
            (0, 4): 6, (1, 4): 7, (2, 4): 8, (3, 4): 9}
    pairs = sorted(pidx)
    r2 = SQ2
    table = {
        ("33", "33"): [-8 / 3, -8 / 3, -2 / 3, -2 / 3, -2 / 3, -2 / 3, -4 / 3, -4 / 3, -4 / 3, -4 / 3],
        ("36", "36"): [-8 / 3, 4 / 3, -5 / 3, -5 / 3, -5 / 3, -5 / 3, 2 / 3, 2 / 3, -10 / 3, -10 / 3],
        ("63", "63"): [4 / 3, -8 / 3, -5 / 3, -5 / 3, -5 / 3, -5 / 3, -10 / 3, -10 / 3, 2 / 3, 2 / 3],
        ("33", "36"): [0, 0, r2, -r2, -r2, r2, 0, 0, -2 * r2, 2 * r2],
        ("33", "63"): [0, 0, -r2, r2, -r2, r2, 2 * r2, -2 * r2, 0, 0],
        ("36", "63"): [0, 0, -1, -1, 1, 1, 0, 0, 0, 0],
    }

    def color_factor(c1, c2, pair):
        key = (c1, c2) if (c1, c2) in table else (c2, c1)
        return table[key][pidx[pair]]

    def spin_factor(s1, s2, pair):
        return 0.25  # all spins aligned, S = 5/2

    c_sign12 = {"33": -1.0, "36": -1.0, "63": 1.0}
    c_sign34 = {"33": -1.0, "36": 1.0, "63": -1.0}
    states = [(c, "s52") for c in colors]
    onehot = [color_onehot[c] + [1] for c, _ in states]
    sym = [
        ("exchange_QQ", [1, 0, 2, 3, 4], False, [-c_sign12[c] for c, _ in states]),
        ("exchange_light", [0, 1, 3, 2, 4], False, [c_sign34[c] for c, _ in states]),
        ("parity", [0, 1, 2, 3, 4], True, [1.0 for _ in states]),
    ]
    return states, onehot, pairs, color_factor, spin_factor, sym, colors, color_keys


def build_system(name: str) -> dict:
    cfg = dict(SYSTEMS[name])
    tables = _tetra_tables() if cfg["kind"] == "tetra" else _penta_tables()
    states, onehot, pairs, color_factor, spin_factor, sym, colors, color_keys = tables
    masses = np.array([QUARK_MASS[p] for p in cfg["particles"]], dtype=np.float64)
    n = len(masses)
    A = len(states)
    P = len(pairs)
    k1 = np.zeros((P, A, A))
    k2 = np.zeros((P, A, A))
    for p, pair in enumerate(pairs):
        for a, (ca, sa) in enumerate(states):
            for b, (cb, sb) in enumerate(states):
                cme = color_factor(ca, cb, pair)
                k1[p, a, b] = -3.0 / 16.0 * cme * (1.0 if sa == sb else 0.0)
                k2[p, a, b] = -3.0 / 16.0 * cme * spin_factor(sa, sb, pair)
    pi_idx = np.array([i for i, _ in pairs])
    pj_idx = np.array([j for _, j in pairs])
    mi, mj = masses[pi_idx], masses[pj_idx]
    r0 = AL1["A"] * (2.0 * mi * mj / (mi + mj)) ** (-AL1["B"])
    hyp = 8.0 * math.pi * AL1["kappa_p"] / (3.0 * mi * mj) / (math.pi ** 1.5 * r0 ** 3)
    cfg.update({
        "name": name,
        "n": n,
        "masses": masses,
        "n_states": A,
        "onehot": np.array(onehot, dtype=np.float64),
        "state_color": np.array([colors.index(c) for c, _ in states]),
        "color_names": [color_keys[c] for c in colors],
        "pi": pi_idx, "pj": pj_idx, "r0": r0, "hyp": hyp, "k1": k1, "k2": k2,
        "mass_sum": float(masses.sum()),
        "sym": sym,
    })
    cfg["dim"] = 3 * n + cfg["onehot"].shape[1]
    return cfg


def potential_matrix(cfg, R):
    """V(R)[a, b] in GeV for configurations R with shape (M, n, 3)."""
    ri = R[:, cfg["pi"], :]
    rj = R[:, cfg["pj"], :]
    r = jnp.sqrt(jnp.sum((ri - rj) ** 2, axis=-1))
    central = -AL1["kappa"] / r + AL1["lam"] * r - AL1["Lambda"]
    hyper = jnp.asarray(cfg["hyp"])[None, :] * jnp.exp(-(r ** 2) / jnp.asarray(cfg["r0"])[None, :] ** 2)
    V = jnp.einsum("mp,pab->mab", central, jnp.asarray(cfg["k1"]))
    V = V + jnp.einsum("mp,pab->mab", hyper, jnp.asarray(cfg["k2"]))
    return V + cfg["mass_sum"] * jnp.eye(cfg["n_states"])[None]


# --------------------------------------------------------------------------
# Submission loading
# --------------------------------------------------------------------------
class SubmissionError(Exception):
    pass


_DTYPE_BYTES = {"f64": 8, "f32": 4, "f16": 2, "bf16": 2, "i64": 8, "i32": 4, "i16": 2, "i8": 1,
                "ui64": 8, "ui32": 4, "ui16": 2, "ui8": 1, "i1": 1}


def _numel(shape_text: str) -> tuple[int, int]:
    parts = shape_text.strip().split("x")
    dims, dtype = parts[:-1], parts[-1]
    numel = 1
    for d in dims:
        numel *= int(d) if d.isdigit() else 1
    if dtype.startswith("complex<"):
        inner = dtype[len("complex<"):-1]
        size = 2 * _DTYPE_BYTES.get(inner, 8)
    else:
        size = _DTYPE_BYTES.get(dtype, 8)
    return numel, size


def count_constants(module_text: str) -> int:
    """Stored constant data of the submitted program, in float64 equivalents.

    Every dense constant contributes its storage in bytes (a splat stores one
    element); the total is divided by 8, so a float64 weight counts as one
    parameter and packing several low-precision values into one wide constant
    does not reduce the count.  Trainable weights baked into an exported
    wavefunction are counted here, as are any other stored tables.
    """
    total_bytes = 0
    for literal, shape in re.findall(r"dense<([^>]*)>\s*:\s*tensor<([^>]*)>", module_text):
        numel, size = _numel(shape)
        lit = literal.strip()
        if lit.startswith('"0x') or lit.startswith('"0X'):
            total_bytes += max(size, (len(lit.strip('"')) - 2) // 2)
        elif lit.startswith("["):
            total_bytes += numel * size
        else:
            total_bytes += size
    for _, shape in re.findall(r"dense_resource<([^>]*)>\s*:\s*tensor<([^>]*)>", module_text):
        numel, size = _numel(shape)
        total_bytes += numel * size
    return int(math.ceil(total_bytes / 8.0))


def load_submission(path: Path, cfg: dict) -> dict:
    if not path.is_file():
        raise SubmissionError(f"missing {path.name}")
    size = path.stat().st_size
    if size == 0 or size > MAX_FILE_BYTES:
        raise SubmissionError(f"bad file size {size}")
    try:
        exp = jexport.deserialize(bytearray(path.read_bytes()))
    except Exception as exc:  # noqa: BLE001
        raise SubmissionError(f"cannot deserialize: {exc!r}") from exc
    if len(exp.in_avals) != 1 or len(exp.out_avals) != 1:
        raise SubmissionError("function must take one array and return one array")
    if exp.ordered_effects or exp.unordered_effects:
        raise SubmissionError("function has side effects")
    if getattr(exp, "nr_devices", 1) != 1:
        raise SubmissionError("function must be single-device")
    platform = jax.default_backend()
    platform = {"gpu": "cuda"}.get(platform, platform)
    if platform not in tuple(exp.platforms):
        raise SubmissionError(f"not exported for platform {platform}; has {exp.platforms}")
    ia, oa = exp.in_avals[0], exp.out_avals[0]
    if len(ia.shape) != 2 or ia.shape[1] != cfg["dim"] or np.dtype(ia.dtype) != np.dtype(np.float64):
        raise SubmissionError(f"input must be float64[batch, {cfg['dim']}], got {ia.dtype}{ia.shape}")
    if len(oa.shape) != 1 or np.dtype(oa.dtype) not in (np.dtype(np.complex128), np.dtype(np.float64)):
        raise SubmissionError(f"output must be complex128[batch] or float64[batch], got {oa.dtype}{oa.shape}")
    text = exp.mlir_module()
    if re.search(r"custom_call\s*@[\w.]*callback", text) or "callback" in " ".join(
            re.findall(r"call_target_name\s*=\s*\"([^\"]*)\"", text)):
        raise SubmissionError("host callbacks are not allowed")
    if re.search(r"stablehlo\.rng|rng_bit_generator|stablehlo\.infeed|stablehlo\.recv", text):
        raise SubmissionError("random-number or input ops are not allowed")
    batch = ia.shape[0]
    if isinstance(batch, (int, np.integer)):
        batch = int(batch)
        if batch <= 0 or CALL_BATCH % batch != 0:
            raise SubmissionError(f"fixed batch {batch} must divide {CALL_BATCH}")
    else:
        batch = None
    return {"exported": exp, "batch": batch, "parameter_count": count_constants(text)}


def make_logpsi(sub: dict, cfg: dict):
    """Return f(rows[CALL_BATCH, dim]) -> complex128[CALL_BATCH]."""
    exp = sub["exported"]
    batch = sub["batch"]
    dim = cfg["dim"]

    def call(x):
        if batch is None:
            out = exp.call(x)
        else:
            out = lax.map(exp.call, x.reshape(CALL_BATCH // batch, batch, dim)).reshape(CALL_BATCH)
        return out.astype(jnp.complex128)

    return call


def make_rows_eval(call, dim):
    """Evaluate log psi on any number of rows, always in CALL_BATCH-row calls.

    Returns (traceable function, jitted function)."""

    def raw(rows):
        m = rows.shape[0]
        pad = (-m) % CALL_BATCH
        if pad:
            rows = jnp.concatenate([rows, jnp.broadcast_to(rows[:1], (pad, dim))], axis=0)
        k = rows.shape[0] // CALL_BATCH
        return lax.map(call, rows.reshape(k, CALL_BATCH, dim)).reshape(-1)[:m]

    return raw, jax.jit(raw)


# --------------------------------------------------------------------------
# Metropolis sampler (owned by the evaluator)
# --------------------------------------------------------------------------
def make_sampler(call, cfg):
    n, A = cfg["n"], cfg["n_states"]
    onehot = jnp.asarray(cfg["onehot"])

    def logabs(R, a):
        x = jnp.concatenate([R, onehot[a]], axis=1)
        lp = jnp.real(call(x))
        return jnp.where(jnp.isnan(lp), -jnp.inf, lp)

    def one_step(state, key):
        R, a, lp, sigma = state
        k1, k2, k3, k4 = jax.random.split(key, 4)
        Rp = R + sigma * jax.random.normal(k1, R.shape)
        lpp = logabs(Rp, a)
        acc1 = jnp.log(jax.random.uniform(k2, lp.shape)) < 2.0 * (lpp - lp)
        acc1 = acc1 & jnp.isfinite(lpp)
        R = jnp.where(acc1[:, None], Rp, R)
        lp = jnp.where(acc1, lpp, lp)
        ap = jax.random.randint(k3, a.shape, 0, A)
        lpp = logabs(R, ap)
        acc2 = (jnp.log(jax.random.uniform(k4, lp.shape)) < 2.0 * (lpp - lp)) & jnp.isfinite(lpp)
        a = jnp.where(acc2, ap, a)
        lp = jnp.where(acc2, lpp, lp)
        return (R, a, lp, sigma), (jnp.mean(acc1), jnp.mean(acc2))

    @jax.jit
    def init(key):
        k1, k2 = jax.random.split(key)
        R = INITIAL_SPREAD * jax.random.normal(k1, (N_CHAINS, 3 * n))
        a = jax.random.randint(k2, (N_CHAINS,), 0, A)
        return R, a, logabs(R, a)

    @jax.jit
    def run_block(R, a, lp, sigma, key):
        keys = jax.random.split(key, ADAPT_BLOCK_STEPS)
        (R, a, lp, _), (acc1, acc2) = lax.scan(one_step, (R, a, lp, sigma), keys)
        return R, a, lp, jnp.mean(acc1), jnp.mean(acc2)

    @jax.jit
    def run_burn(R, a, lp, sigma, key):
        keys = jax.random.split(key, EXTRA_BURN_STEPS)
        (R, a, lp, _), (acc1, acc2) = lax.scan(one_step, (R, a, lp, sigma), keys)
        return R, a, lp, jnp.mean(acc1), jnp.mean(acc2)

    @jax.jit
    def run_samples(R, a, lp, sigma, key):
        def outer(state, k):
            ks = jax.random.split(k, THIN)
            state, (acc1, acc2) = lax.scan(one_step, state, ks)
            return state, (state[0], state[1], state[2], jnp.mean(acc1), jnp.mean(acc2))

        keys = jax.random.split(key, SAMPLES_PER_CHAIN)
        state, (Rs, as_, lps, acc1, acc2) = lax.scan(outer, (R, a, lp, sigma), keys)
        return Rs, as_, lps, jnp.mean(acc1), jnp.mean(acc2)

    def sample(key):
        k_init, k_adapt, k_burn, k_samp = jax.random.split(key, 4)
        R, a, lp = init(k_init)
        sigma = INITIAL_STEP
        adapt_keys = jax.random.split(k_adapt, ADAPT_BLOCKS)
        for i in range(ADAPT_BLOCKS):
            R, a, lp, acc1, _ = run_block(R, a, lp, sigma, adapt_keys[i])
            acc1 = float(acc1)
            sigma = float(np.clip(sigma * math.exp(acc1 - 0.5), 1e-4, 50.0))
        R, a, lp, acc_b1, acc_b2 = run_burn(R, a, lp, sigma, k_burn)
        Rs, as_, lps, acc1, acc2 = run_samples(R, a, lp, sigma, k_samp)
        stats = {"step_size": sigma, "accept_spatial": float(acc1), "accept_discrete": float(acc2),
                 "accept_spatial_burn": float(acc_b1)}
        return Rs, as_, lps, stats  # (S, C, 3n), (S, C), (S, C)

    return sample


# --------------------------------------------------------------------------
# Local energy
# --------------------------------------------------------------------------
def _rotations(key, shape):
    q = jax.random.normal(key, shape + (4,))
    q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    row0 = jnp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1)
    row1 = jnp.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1)
    row2 = jnp.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)
    return jnp.stack([row0, row1, row2], -2)


def _expm1c(d):
    dr, di = jnp.real(d), jnp.imag(d)
    re_part = jnp.expm1(dr) * jnp.cos(di) - 2.0 * jnp.sin(di / 2.0) ** 2
    im_part = jnp.exp(dr) * jnp.sin(di)
    return re_part + 1j * im_part


def make_energy_fns(cfg, call, rows_raw):
    """Local energies E_L = (H psi)(R, a) / psi(R, a) of sampled configurations.

    For each sample the rows are: the configuration itself, 6n finite-difference
    stencil points (randomly rotated axes, random step), and the same coordinates
    in every discrete channel (in a random per-sample order).  In the main pass
    the (M, K) rows of M = CALL_BATCH samples are transposed so that each call of
    the submission holds one row of each of M different samples.
    """
    n, A = cfg["n"], cfg["n_states"]
    onehot = jnp.asarray(cfg["onehot"])
    masses = jnp.asarray(cfg["masses"])
    K = 1 + 6 * n + A
    dim = cfg["dim"]

    def build_rows(R, a, key):
        """R (M, n, 3), a (M,) -> rows (M, K, dim), h (M,), order (M, A)."""
        M = R.shape[0]
        kh, kq, ko = jax.random.split(key, 3)
        h = jax.random.uniform(kh, (M,), minval=FD_H_MIN, maxval=FD_H_MAX)
        Q = _rotations(kq, (M, n))                      # (M, n, 3, 3); columns are axes
        axes = jnp.swapaxes(Q, -1, -2) * h[:, None, None, None]   # (M, k, d, xyz)
        sel = jnp.eye(n)                                 # (k, particle)
        signs = jnp.array([1.0, -1.0])
        delta = (sel[None, :, None, None, :, None] * axes[:, :, :, None, None, :]
                 * signs[None, None, None, :, None, None])            # (M, k, d, s, n, 3)
        Rst = (R[:, None, None, None, :, :] + delta).reshape(M, 6 * n, 3 * n)
        center = R.reshape(M, 1, 3 * n)
        pos = jnp.concatenate([center, Rst, jnp.broadcast_to(center, (M, A, 3 * n))], axis=1)
        order = jnp.argsort(jax.random.uniform(ko, (M, A)), axis=1)  # random channel order per sample
        own = onehot[a][:, None, :]
        disc = jnp.concatenate([own, jnp.broadcast_to(own, (M, 6 * n, own.shape[-1])), onehot[order]], axis=1)
        rows = jnp.concatenate([pos, disc], axis=-1)
        return rows, h, order

    def energy_from(L, R, a, h, order):
        """L (M, K) log psi of the rows -> local energy, kinetic part, Re log psi(R, a)."""
        M = R.shape[0]
        l0 = L[:, 0]
        lst = L[:, 1:1 + 6 * n].reshape(M, n, 3, 2)
        ldisc = L[:, 1 + 6 * n:]
        rst = _expm1c(lst - l0[:, None, None, None])
        lap = jnp.sum(rst[..., 0] + rst[..., 1], axis=-1) / (h[:, None] ** 2)   # (M, n)
        kinetic = -jnp.sum(lap / (2.0 * masses[None, :]), axis=-1)
        rdisc = jnp.exp(ldisc - l0[:, None])
        V = potential_matrix(cfg, R)
        Va = V[jnp.arange(M), a, :]
        Vo = jnp.take_along_axis(Va, order, axis=1)
        pot = jnp.sum(Vo * rdisc, axis=-1)
        return jnp.real(kinetic + pot), jnp.real(kinetic), jnp.real(l0)

    def eval_transposed(rows):
        rows_t = jnp.swapaxes(rows, 0, 1)            # (K, M, dim), M == CALL_BATCH
        return jnp.swapaxes(lax.map(call, rows_t), 0, 1)

    def eval_natural(rows):
        M = rows.shape[0]
        return rows_raw(rows.reshape(M * K, dim)).reshape(M, K)

    def one_chunk(args):
        R, a, key = args
        rows, h, order = build_rows(R, a, key)
        return energy_from(eval_transposed(rows), R, a, h, order)

    @jax.jit
    def energy_block(Rb, ab, keys):
        """Rb (G, M, n, 3), ab (G, M), keys (G, 2) -> E_L, kinetic, Re log psi, each (G, M)."""
        return lax.map(one_chunk, (Rb, ab, keys))

    @jax.jit
    def energy_natural(R, a, key):
        """Same chunk in the natural (sample-major) call layout."""
        rows, h, order = build_rows(R, a, key)
        return energy_from(eval_natural(rows), R, a, h, order)

    return energy_block, energy_natural, K


# --------------------------------------------------------------------------
# Scoring one submission
# --------------------------------------------------------------------------
def _split_rhat(x):
    """x: (S, C) local energies -> split R-hat over chains."""
    S = x.shape[0] // 2
    chains = np.concatenate([x[:S].T, x[S:2 * S].T], axis=0)   # (2C, S)
    means = chains.mean(axis=1)
    W = chains.var(axis=1, ddof=1).mean()
    B = S * means.var(ddof=1)
    var_plus = (S - 1) / S * W + B / S
    return float(np.sqrt(var_plus / W)) if W > 0 else float("inf")


def _close(x, y, abs_tol, rel_tol):
    """Elementwise |x - y| <= abs_tol + rel_tol |y|, treating equal infinities as equal."""
    x = np.asarray(x)
    y = np.asarray(y)
    same = (x == y)
    with np.errstate(invalid="ignore", over="ignore"):
        ok = np.abs(x - y) <= abs_tol + rel_tol * np.abs(y)
    return same | ok


def _logpsi_dist(x, y):
    """Max distance between two complex log psi arrays (phase modulo 2 pi); equal -inf counts as 0."""
    x = np.asarray(x, dtype=np.complex128)
    y = np.asarray(y, dtype=np.complex128)
    both_zero = np.isneginf(x.real) & np.isneginf(y.real)
    with np.errstate(invalid="ignore"):
        d = x - y
        dist = np.maximum(np.abs(d.real), 2.0 * np.abs(np.sin(d.imag / 2.0)))
    dist = np.where(both_zero, 0.0, dist)
    dist = np.where(np.isnan(dist), np.inf, dist)
    return float(np.max(dist)) if dist.size else 0.0


def evaluate_submission(path: Path, cfg: dict, seed: int, chain_means_out=None) -> dict:
    t0 = time.time()
    info = {"file": str(path), "gates": {}}
    sub = load_submission(path, cfg)
    info["parameter_count"] = sub["parameter_count"]
    info["fixed_batch"] = sub["batch"]
    if sub["parameter_count"] > PARAM_BUDGET:
        raise SubmissionError(f"parameter count {sub['parameter_count']} exceeds budget {PARAM_BUDGET}",
                              info)

    call = make_logpsi(sub, cfg)
    rows_raw, rows_eval = make_rows_eval(call, cfg["dim"])
    sample = make_sampler(call, cfg)
    energy_block, energy_natural, K = make_energy_fns(cfg, call, rows_raw)

    system_index = list(SYSTEMS).index(cfg["name"])
    base = jax.random.fold_in(jax.random.PRNGKey(seed), system_index)
    k_samp, k_energy, _ = jax.random.split(base, 3)

    Rs, as_, lps, stats = sample(k_samp)
    info.update(stats)
    info["seconds_sampling"] = time.time() - t0
    S, C = as_.shape
    n = cfg["n"]
    R_all = np.asarray(Rs).reshape(S * C, n, 3)
    a_all = np.asarray(as_).reshape(S * C)
    lp_all = np.asarray(lps).reshape(S * C)
    del Rs, as_, lps
    N = S * C

    # local energies: ENERGY_CHUNK samples per chunk, ENERGY_GROUP chunks per compiled call
    n_chunks = N // ENERGY_CHUNK
    keys = jax.vmap(lambda c: jax.random.fold_in(k_energy, c))(jnp.arange(n_chunks))
    e_loc = np.empty(N)
    kin = np.empty(N)
    l0_all = np.empty(N)
    block = ENERGY_GROUP * ENERGY_CHUNK
    for g in range(N // block):
        sl = slice(g * block, (g + 1) * block)
        Rb = jnp.asarray(R_all[sl].reshape(ENERGY_GROUP, ENERGY_CHUNK, n, 3))
        ab = jnp.asarray(a_all[sl].reshape(ENERGY_GROUP, ENERGY_CHUNK))
        e, t, l0 = energy_block(Rb, ab, keys[g * ENERGY_GROUP:(g + 1) * ENERGY_GROUP])
        e_loc[sl] = np.asarray(e).reshape(-1)
        kin[sl] = np.asarray(t).reshape(-1)
        l0_all[sl] = np.asarray(l0).reshape(-1)
    info["seconds_energy"] = time.time() - t0
    finite = np.isfinite(e_loc)
    if finite.any():
        info["energy_mev_unchecked"] = float(e_loc[finite].mean() * 1000.0)

    # gate: finiteness
    if not np.all(np.isfinite(e_loc)):
        raise SubmissionError(f"{int(np.sum(~np.isfinite(e_loc)))} non-finite local energies", info)

    # gate: the function is deterministic for identical calls
    probe = jnp.concatenate([jnp.asarray(R_all[:CALL_BATCH].reshape(CALL_BATCH, -1)),
                             jnp.asarray(cfg["onehot"])[jnp.asarray(a_all[:CALL_BATCH])]], axis=1)
    det = _logpsi_dist(np.asarray(rows_eval(probe)), np.asarray(rows_eval(probe)))
    info["gates"]["repeat_determinism"] = det
    if not det <= DET_TOL:
        raise SubmissionError(f"log psi changes between identical calls (max diff {det:.3e})", info)

    # gate: the sampler and the energy pass saw the same wavefunction
    consist = np.abs(l0_all - lp_all)
    consist = np.where((l0_all == lp_all), 0.0, consist)
    consist = float(np.max(np.where(np.isnan(consist), np.inf, consist)))
    info["gates"]["sampler_energy_consistency"] = consist
    if not consist <= CONSIST_TOL:
        raise SubmissionError(f"log|psi| differs between sampling and energy calls (max diff {consist:.3e})",
                              info)

    # gate: local energies do not depend on how rows are grouped into calls
    e_nat, t_nat, l0_nat = energy_natural(jnp.asarray(R_all[:ENERGY_CHUNK]), jnp.asarray(a_all[:ENERGY_CHUNK]),
                                          keys[0])
    e_nat = np.asarray(e_nat)
    ok = _close(e_nat, e_loc[:ENERGY_CHUNK], LAYOUT_ABS_TOL, LAYOUT_REL_TOL)
    ok &= _close(np.asarray(l0_nat), l0_all[:ENERGY_CHUNK], CONSIST_TOL, 0.0)
    with np.errstate(invalid="ignore"):
        lay = np.abs(e_nat - e_loc[:ENERGY_CHUNK])
    info["gates"]["layout_max_energy_diff_gev"] = float(np.max(np.where(np.isfinite(lay), lay, 0.0)))
    info["gates"]["layout_mismatches"] = int(np.sum(~ok))
    if not np.all(ok):
        raise SubmissionError(f"local energies depend on the call layout ({int(np.sum(~ok))} of {ENERGY_CHUNK} "
                              f"samples differ)", info)

    # gate: physical sector (Fermi statistics with isospin, parity)
    idx = np.linspace(0, N - 1, SYM_SAMPLES).astype(np.int64)
    Rsym = R_all[idx]
    asym = a_all[idx]
    onehot = cfg["onehot"]
    base_rows = np.concatenate([Rsym.reshape(len(idx), -1), onehot[asym]], axis=1)
    l0 = np.asarray(rows_eval(jnp.asarray(base_rows)))
    violations = {}
    for name, perm_p, invert, eta in cfg["sym"]:
        Rg = Rsym[:, perm_p, :] * (-1.0 if invert else 1.0)
        rows_g = np.concatenate([Rg.reshape(len(idx), -1), onehot[asym]], axis=1)
        lg = np.asarray(rows_eval(jnp.asarray(rows_g)))
        with np.errstate(over="ignore", invalid="ignore"):
            ratio = np.exp(lg - l0)
        target = np.asarray(eta)[asym]
        v = float(np.mean(np.abs(ratio - target) ** 2))
        violations[name] = v if np.isfinite(v) else float("inf")
    info["gates"]["symmetry"] = violations
    worst = max(violations.values())
    if not np.isfinite(worst) or worst > SYM_TOL:
        raise SubmissionError(f"wavefunction outside the physical sector (violation {violations})", info)

    # statistics
    x = e_loc.reshape(S, C)
    chain_means = x.mean(axis=0)
    energy = float(e_loc.mean())
    mc_error = float(chain_means.std(ddof=1) / math.sqrt(C))
    rhat = _split_rhat(x)
    info["r_hat"] = rhat
    if chain_means_out is not None:
        chain_means_out[cfg["name"]] = (chain_means * 1000.0).tolist()
    if not np.isfinite(rhat) or rhat > RHAT_MAX:
        raise SubmissionError(f"sampler did not converge (split R-hat {rhat:.3f})", info)

    energy_mev = energy * 1000.0
    binding_raw = energy_mev - cfg["threshold_mev"]
    info["energy_mev"] = energy_mev
    info["binding_raw_mev"] = binding_raw
    info["mc_error_raw_mev"] = mc_error * 1000.0
    info["kinetic_mev"] = float(kin.mean() * 1000.0)
    info["local_energy_std_mev"] = float(e_loc.std() * 1000.0)
    info["symmetry_violation"] = worst
    if binding_raw < cfg["floor_mev"]:
        raise SubmissionError(f"energy {binding_raw:.1f} MeV below threshold is physically implausible", info)

    bound = binding_raw < 0.0
    info["bound"] = bool(bound)
    colors = np.bincount(cfg["state_color"][a_all], minlength=len(cfg["color_names"])) / N
    scores = {
        # scored: binding below the fixed threshold (0 = no bound state) and its Monte Carlo error;
        # a run without a bound state has failed the task and gets the worst value of both
        "binding_mev": binding_raw if bound else 0.0,
        "mc_error_mev": min(mc_error * 1000.0, MC_ERROR_CAP_MEV) if bound else MC_ERROR_CAP_MEV,
        # diagnostics (not ranked)
        "energy_mev": energy_mev,
        "parameter_count": float(sub["parameter_count"]),
        "valid": 1.0,
    }
    for key, value in zip(cfg["color_names"], colors):
        scores[key] = float(value)
    info["seconds"] = time.time() - t0
    return {"scores": scores, "info": info}


def failed_scores(cfg: dict) -> dict:
    """Worst value of every scored metric."""
    return {"binding_mev": 0.0, "mc_error_mev": MC_ERROR_CAP_MEV, "valid": 0.0}


def baseline_scores(cfg: dict) -> dict:
    """Fixed no-binding baseline: the state is placed at the two-hadron threshold."""
    return {"binding_mev": 0.0, "mc_error_mev": MC_ERROR_CAP_MEV, "energy_mev": cfg["threshold_mev"]}


def discover_methods(artifact_root: Path) -> list[str]:
    methods = set()
    if not artifact_root.is_dir():
        return []
    for cfg in SYSTEMS.values():
        suffix = "_" + cfg["suffix"]
        for d in artifact_root.iterdir():
            if d.is_dir() and d.name.endswith(suffix) and len(d.name) > len(suffix):
                methods.add(d.name[: -len(suffix)])
    methods.discard("threshold_baseline")
    return sorted(methods)


def _score_one(path: Path, exp_name: str, seed: int) -> dict:
    """Score one submission in this process; never raises."""
    cfg = build_system(exp_name)
    chain_means = {}
    try:
        out = evaluate_submission(path, cfg, seed, chain_means)
        return {"status": "valid", "scores": out["scores"], "info": out["info"],
                "chain_means": chain_means.get(exp_name), "backend": jax.default_backend()}
    except Exception as exc:  # noqa: BLE001  -- any failure scores as failed
        info = exc.args[1] if isinstance(exc, SubmissionError) and len(exc.args) > 1 else {}
        return {"status": "failed", "reason": (str(exc.args[0]) if exc.args else repr(exc))[:2000],
                "trace": traceback.format_exc(limit=3) if not isinstance(exc, SubmissionError) else "",
                "info": {k: v for k, v in info.items() if k != "file"}}


def _score_isolated(path: Path, exp_name: str, seed: int) -> dict:
    """Score one submission in a fresh process so a crash or hang only fails that submission."""
    with tempfile.TemporaryDirectory() as tmp:
        result = Path(tmp) / "result.json"
        cmd = [sys.executable, os.path.abspath(__file__), "--one", str(path), "--exp", exp_name,
               "--seed", str(seed), "--result", str(result)]
        try:
            proc = subprocess.run(cmd, timeout=SUBMISSION_TIMEOUT, capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            return {"status": "failed", "reason": f"evaluation exceeded {SUBMISSION_TIMEOUT} s", "info": {}}
        if result.is_file():
            try:
                return json.loads(result.read_text())
            except json.JSONDecodeError:
                pass
        return {"status": "failed", "reason": f"evaluator process exited with code {proc.returncode}",
                "stderr_tail": (proc.stderr or "")[-3000:], "info": {}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--artifacts", default="checkpoints")
    parser.add_argument("--output", default=f"scoring/scores_{SLICE}.json")
    parser.add_argument("--seed", type=int, default=SEED, help="sampler seed (audit use only)")
    parser.add_argument("--chain-means", default=None, help="optional JSON path for per-chain energy means")
    parser.add_argument("--one", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--exp", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--result", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.one is not None:   # worker mode: one submission, one system
        res = _score_one(Path(args.one), args.exp, args.seed)
        Path(args.result).write_text(json.dumps(res, default=float) + "\n")
        return

    artifact_root = Path(args.artifacts)
    methods = discover_methods(artifact_root)
    print(f"[{SLICE}] methods={methods}", flush=True)

    scored = {"slice": SLICE, "experiments": {}}
    details = {"slice": SLICE, "protocol": {
        "chains": N_CHAINS, "samples_per_chain": SAMPLES_PER_CHAIN, "thin": THIN,
        "parameter_budget": PARAM_BUDGET, "symmetry_tolerance": SYM_TOL,
        "repeat_tolerance": DET_TOL, "layout_tolerance_gev": LAYOUT_ABS_TOL,
        "consistency_tolerance": CONSIST_TOL, "rhat_max": RHAT_MAX,
        "mc_error_cap_mev": MC_ERROR_CAP_MEV}, "submissions": {}}
    chain_means = {}
    for exp_name in SYSTEMS:
        cfg = build_system(exp_name)
        results = {"threshold_baseline": baseline_scores(cfg)}
        for method in methods:
            path = artifact_root / f"{method}_{cfg['suffix']}" / FILE_NAME
            key = f"{exp_name}/{method}"
            res = _score_isolated(path, exp_name, args.seed)
            if res.get("status") == "valid":
                results[method] = res["scores"]
                details["submissions"][key] = {"status": "valid", "backend": res.get("backend"), **res["info"]}
                if res.get("chain_means") is not None:
                    chain_means.setdefault(method, {})[exp_name] = res["chain_means"]
            else:
                results[method] = failed_scores(cfg)
                details["submissions"][key] = {
                    "status": "failed", "reason": res.get("reason", "unknown"), "trace": res.get("trace", ""),
                    **({"stderr_tail": res["stderr_tail"]} if res.get("stderr_tail") else {}),
                    **res.get("info", {})}
            row = results[method]
            print(f"[{SLICE}] {exp_name} {method}: binding={row['binding_mev']:.3f} MeV "
                  f"valid={row.get('valid', 0)} {details['submissions'][key].get('reason', '')}", flush=True)
        scored["experiments"][exp_name] = {"results": results}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(scored, indent=2) + "\n")
    det_path = out_path.with_name(out_path.stem + "_details.json")
    det_path.write_text(json.dumps(details, indent=2, default=float) + "\n")
    if args.chain_means:
        Path(args.chain_means).write_text(json.dumps(chain_means) + "\n")


if __name__ == "__main__":
    main()
