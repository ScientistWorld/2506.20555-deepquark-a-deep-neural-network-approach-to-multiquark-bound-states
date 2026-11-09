import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['NETKET_EXPERIMENTAL_SHARDING'] = '1'

import flax
import jax
import json
import netket as nk

from model_ import QQqq_FCN, Q1Q2qq_FCN, exchange_indices, Penta_FCN
from sampler_ import GaussianFlipRule
from config_ import quarks, MASS, I, default_serializer, out_path, nlayers, nhid, bound, sigma, n
from hi_op_ import _nd, _nparticles, hi, hi_dis, n_discrete, ha


_mcstate_counter = 0


def make_mcstate(*args, **kwargs):
    """nk.vqs.MCState, seeded from DQ_SEED when it is set (otherwise unseeded)."""
    global _mcstate_counter
    _mcstate_counter += 1
    seed = os.environ.get("DQ_SEED")
    if seed is not None and seed != "":
        s = int(seed) + 1009 * _mcstate_counter
        kwargs.setdefault("seed", s)
        kwargs.setdefault("sampler_seed", s + 500003)
    return nk.vqs.MCState(*args, **kwargs)


def env_int(name, default):
    return int(os.environ.get(name, default))


def env_float(name, default):
    return float(os.environ.get(name, default))


def build_model():
    if quarks[0] == quarks[1] and quarks[2] == quarks[3]:
        idparticles = [[0, 1, -1], [2, 3, (-1) ** I]]
    elif quarks[2] == quarks[3]:
        idparticles = [[2, 3, (-1) ** I]]
    else:
        raise ValueError("Not implemented case of identical particles")

    swap_indices, signs = exchange_indices(_nparticles, identical_particles=idparticles)
    if _nparticles == 4:
        if quarks[0] == quarks[1]:
            return QQqq_FCN(nd=_nd, nparticles=_nparticles, mass=tuple(MASS), ndiscrete=n_discrete,
                            nhid=nhid, nlayers=nlayers, bound=bound, indices=swap_indices, signs=signs)
        return Q1Q2qq_FCN(nd=_nd, nparticles=_nparticles, mass=tuple(MASS), ndiscrete=n_discrete,
                          nhid=nhid, nlayers=nlayers, bound=bound, indices=swap_indices, signs=signs)
    if _nparticles == 5:
        return Penta_FCN(nd=_nd, nparticles=_nparticles, mass=tuple(MASS), ndiscrete=n_discrete,
                         nhid=nhid, nlayers=nlayers, bound=bound, indices=swap_indices, signs=signs)
    raise ValueError(f"Unsupported particle count: {_nparticles}")


def main():
    if n is None:
        raise ValueError("Pass --n with the run directory to continue, e.g. --n 1")

    logpath = out_path('eval', n)
    existing = []
    for entry in os.listdir(logpath):
        if entry.startswith("Train3_") and entry.endswith(".mpack"):
            try:
                existing.append(int(entry[len("Train3_"):-len(".mpack")]))
            except ValueError:
                pass
    if not existing:
        raise FileNotFoundError(f"No Train3_*.mpack checkpoints found in {logpath}")

    max_existing = max(existing)
    start = env_int("DQ_RESUME_FROM", max_existing)
    if start not in existing:
        raise FileNotFoundError(f"Requested DQ_RESUME_FROM={start}, but Train3_{start}.mpack was not found in {logpath}")

    first_new = env_int("DQ_RESUME_START_INDEX", max_existing + 1)
    blocks = env_int("DQ_RESUME_TRAIN3_BLOCKS", 10)
    n_iter = env_int("DQ_RESUME_TRAIN3_ITERS", env_int("DQ_TRAIN3_ITERS", 50))
    lr = env_float("DQ_RESUME_LR", 0.001)
    nchains = env_int("DQ_NCHAINS", 2000)
    nsweep = env_int("DQ_NSWEEP", 20)
    nsample = env_int("DQ_NSAMPLE", 2 * 10 ** 4)
    ndiscard = env_int("DQ_TRAIN2_NDISCARD", 200)

    model = build_model()
    sampler = nk.sampler.MetropolisSampler(
        hi, GaussianFlipRule(hi_dis=hi_dis, sigma=sigma), n_chains=nchains, sweep_size=nsweep
    )
    vs = make_mcstate(
        sampler, model, n_samples=nsample, n_discard_per_chain=ndiscard, chunk_size=max(1, nsample // 4)
    )
    with open(f"{logpath}/Train3_{start}.mpack", "rb") as handle:
        vs.variables = flax.serialization.from_bytes(vs.variables, handle.read())

    gs = nk.VMC(ha, nk.optimizer.Sgd(lr), variational_state=vs, preconditioner=nk.optimizer.SR())
    for i in range(first_new, first_new + blocks):
        if os.path.exists(f"{logpath}/Train3_{i}.mpack") and os.environ.get("DQ_RESUME_OVERWRITE", "0") != "1":
            raise FileExistsError(
                f"Refusing to overwrite Train3_{i}.mpack. Set DQ_RESUME_START_INDEX above existing checkpoints "
                "or DQ_RESUME_OVERWRITE=1."
            )
        print(f"continue_train3_{i}", flush=True)
        gs.run(n_iter=n_iter, out=f"{logpath}/Train3_{i}")

    train_path = f"{logpath}/train.json"
    with open(train_path, "r", encoding="utf-8") as handle:
        train = json.load(handle)
    train["Train3_continued"] = {
        "from_checkpoint": start,
        "max_existing_checkpoint_before_resume": max_existing,
        "first_new_checkpoint": first_new,
        "blocks": blocks,
        "n_iter": n_iter,
        "nchains": nchains,
        "nsweep": nsweep,
        "nsample": nsample,
        "ndiscard": ndiscard,
        "lr": lr,
        "sigma": sigma * 0.197,
    }
    with open(train_path, "w", encoding="utf-8") as handle:
        json.dump(train, handle, indent=2, default=default_serializer, ensure_ascii=False)


if __name__ == "__main__":
    main()
