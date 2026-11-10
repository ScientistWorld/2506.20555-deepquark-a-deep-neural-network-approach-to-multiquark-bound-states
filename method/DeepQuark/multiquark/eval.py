import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['NETKET_EXPERIMENTAL_SHARDING'] = '1'

import netket as nk
import jax, flax
import sys
import json
import os

from model_ import QQqq_FCN, Q1Q2qq_FCN, exchange_indices, Penta_FCN
from sampler_ import GaussianFlipRule
from config_ import quarks, MASS, S, I, default_serializer, out_path, nlayers, nhid, bound, sigma, n, irange
from operator_ import MultiPotentialEnergy
from hi_op_ import (_nd, _nparticles, hi, hi_dis, n_discrete, dof_color, ha,
                    color_proportions, spin_proportions, ms_radius)

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


"""Model"""
if quarks[0] == quarks[1] and quarks[2] == quarks[3]:
    idparticles = [[0, 1, -1], [2, 3, (-1) ** I]]
elif quarks[2] == quarks[3]:
    idparticles = [[2, 3, (-1) ** I]]
else:
    raise ValueError("Not implemented case of identical particles")


swap_indices, signs = exchange_indices(_nparticles, identical_particles=idparticles)

if _nparticles == 4:
    if quarks[0] == quarks[1]:
        model = QQqq_FCN(nd=_nd, nparticles=_nparticles, mass=tuple(MASS), ndiscrete=n_discrete, nhid=nhid,
                         nlayers=nlayers,
                         bound=bound, indices=swap_indices, signs=signs)
    else:
        model = Q1Q2qq_FCN(nd=_nd, nparticles=_nparticles, mass=tuple(MASS), ndiscrete=n_discrete, nhid=nhid,
                           nlayers=nlayers,
                           bound=bound, indices=swap_indices, signs=signs)
elif _nparticles == 5:
    model = Penta_FCN(nd=_nd, nparticles=_nparticles, mass=tuple(MASS), ndiscrete=n_discrete, nhid=nhid,
                      nlayers=nlayers, bound=bound, indices=swap_indices, signs=signs)

def env_int(name, default):
    return int(os.environ.get(name, default))


def single_spin_proportion(spin):
    try:
        return spin[0]
    except TypeError:
        return spin


nchains = env_int("DQ_EVAL_NCHAINS", 2000)
nsweep = env_int("DQ_EVAL_NSWEEP", 100)
nsample = env_int("DQ_EVAL_NSAMPLE", 4 * 10 ** 5)
ndiscard = env_int("DQ_EVAL_NDISCARD", 1000)
skip_rms = os.environ.get("DQ_EVAL_SKIP_RMS", "0") == "1"
evaluation = {'nchains': nchains, 'nsweep': nsweep, 'nsample': nsample, 'ndiscard': ndiscard, 'sigma': sigma * 0.197}
if skip_rms:
    evaluation['skip_rms'] = True
sa = nk.sampler.MetropolisSampler(hi, GaussianFlipRule(hi_dis=hi_dis, sigma=sigma), n_chains=nchains,
                                  sweep_size=nsweep)
eval_vs = make_mcstate(sa, model, n_samples=nsample, n_discard_per_chain=ndiscard,
                       chunk_size=env_int("DQ_EVAL_CHUNK", max(1, nsample // 4)))

logpath = out_path('eval', n)
with open(f'{logpath}/eval.json', 'w', encoding='utf-8') as f:
    json.dump(evaluation, f, indent=2, default=default_serializer, ensure_ascii=False)

"""Eval"""
results = {}
for i in irange:
    print(f"para_{i}", flush=True)
    with open(f"{logpath}/Train3_{i}.mpack", 'rb') as file:
        variables = flax.serialization.from_bytes(eval_vs.variables, file.read())
    eval_vs.variables = variables
    results[i] = {}
    for _ in range(env_int("DQ_EVAL_REPEATS", 2)):
        e = eval_vs.expect(ha)

        samples = eval_vs.samples
        color_samples = samples[:, :, -n_discrete:-n_discrete + dof_color]
        spin_samples = samples[:, :, -n_discrete + dof_color:]
        color = color_proportions(color_samples)
        spin = spin_proportions(spin_samples)

        if _nparticles == 4:
            colorresult = f"3*3:{color[0]:.3f},6*6:{color[1]:.3f}"
            if S == 0:
                spinresult = f"0*0:{spin[0]:.3f},1*1:{spin[1]:.3f}"
            elif S == 1:
                spinresult = f"0*1:{spin[0]:.3f},1*0:{spin[1]:.3f},1*1:{spin[2]:.3f}"
            else:
                spinresult = f"1*1:{single_spin_proportion(spin):.3f}"
        elif _nparticles == 5:
            colorresult = f"3*3:{color[0]:.3f},3*6:{color[1]:.3f},6*3:{color[2]:.3f}"
            if S == 1 / 2:
                spinresult = f"0*0->0:{spin[0]:.3f},1*1->0:{spin[1]:.3f},0*1->1:{spin[2]:.3f},1*0->1:{spin[3]:.3f},1*1->1:{spin[4]:.3f}"
            elif S == 3 / 2:
                spinresult = f"0*1->1:{spin[0]:.3f},1*0->1:{spin[1]:.3f},1*1->1:{spin[2]:.3f},1*1->2:{spin[3]:.3f}"
            else:
                spinresult = f"1*1->2:{single_spin_proportion(spin):.3f}"

        rmsresult = ""
        if not skip_rms:
            for l in range(_nparticles - 1):
                for q in range(l + 1, _nparticles):
                    ms = MultiPotentialEnergy(hi, hi_dis, lambda a, a_, r: ms_radius(l, q, a, a_, r))
                    rij2 = eval_vs.expect(ms)
                    rmsresult += f"r{l + 1}{q + 1}: {jax.numpy.sqrt(rij2.mean.real):.3f} "

        results[i][_ + 1] = {'Emean': f"{float(e.mean.real):.5f}", 'Evar': f"{float(e.variance):.5f}",
                             'Esigma': f"{float(e.error_of_mean):.5f}", 'ERhat': f"{float(e.R_hat):.3f}",
                             'color': colorresult, 'spin': spinresult, 'rms': rmsresult}
        eval_vs.reset()

with open(f'{logpath}/results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, default=default_serializer, ensure_ascii=False)
