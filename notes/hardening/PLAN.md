# Hardening iteration — working plan (delete or fold into HARDENING.md when done)

## Design (implemented before job submission)
- Submission = `checkpoints/<method>_<sys>/wavefunction.jaxexport` (jax.export of log psi,
  float64[batch, dim] -> complex128/float64[batch]). Evaluators ignore every JSON a method writes.
- `eval/{train,test}/evaluate.py` (separate copies; test: seed 918273645, 512 samples/chain;
  train: seed 12345, 128 samples/chain; 4096 chains) own: AL1 Hamiltonian, sector (exchange
  + parity gates), thresholds, Metropolis sampler, finite-difference local energy.
- Gates -> failed: missing/corrupt, side effects/callbacks/RNG ops, param budget 10000
  (bytes/8 of stored constants), non-finite E_L, repeat determinism, sampler-vs-energy
  log|psi| consistency (1e-8), natural-vs-transposed layout (1e-6 GeV), symmetry (1e-6),
  split R-hat <= 1.2, floor (Tcc -100, Tbb -400, Pcc -100 MeV).
- Energy pass uses a transposed layout: each call holds one row of each of 4096 samples,
  discrete-channel probes in random per-sample order -> stencils cannot be recognised.
- Each submission runs in its own subprocess (timeout 3600 s) -> crash/hang = failed only.
- Scored: binding_mev (coef 1.0, = min(E-thr,0)), mc_error_mev (coef 0.1, capped 5 MeV).
  Failed OR unbound -> binding 0, mc_error 5 (worst). energy/params/colors/valid = coef 0.
- Experiments: tcc (0.45), tbb (0.35), pcc (0.2) — revisit weights after noise numbers.

## Job (scripts/hardening/driver.py, resumable via tmp/hardening/state/*.done)
- Job d5c89f5c (2 GPUs, 3 concurrent tracks) cancelled at 40 min: container lacked
  `flatbuffers` (jax.export serialization) -> added to setup.sh + job-only fallback
  tmp/pydeps in run.sh; and 3 JAX processes on the job's single CPU ran ~12x slower
  (4 s/iter vs 0.33 s/iter) -> driver is now sequential, 1 GPU.
- Plan order (stage skipped if its estimate does not fit the remaining 5.75 h; rerun the
  same job to continue): old-reference exports + eval + NetKet crosscheck; ref_ccqqc,
  ref_ccqq, ref_bbqq; official_scoring; redteam exports + evals; seed1_ccqqc, better_ccqqc,
  seed1_ccqq, better_ccqq, their evals; shortcuts; evaluator-seed noise; better_bbqq.
- Results: notes/hardening/results/*.json (+ *_details.json), crosscheck_*.json, job_state.json.
  Logs: tmp/hardening/logs/.

## After the job
1. Check crosscheck (gym vs NetKet local energies), timings (info.seconds), r_hat, gates.
2. Before/after table: old JSON evaluator (REDTEAM numbers) vs new evaluator.
3. Seed noise (ref vs seed1 vs rt seeds), evaluator noise (seeds 101/202), better vs ref.
4. Decide weights; write HARDENING.md, update briefing/, scoring/*.md, DETAILS? CONSTRAINTS.
5. Restore scripts/run.sh to reproduce the reference end to end (scripts/reproduce.sh).
6. download.sh: no external data (code vendored); keep idempotent mkdir.
7. python validate.py; commit; action done.

## Observations from job 586a8748 (first real evaluator run)
- NetKet crosscheck (Tcc, 16k configs): mean diff -3e-5 MeV, median |diff| 8e-5 MeV -> evaluator correct.
- Evaluator ~30-37 s per system (test). Old reference (test): Tcc -13.27 (err 0.15), Tbb -152.63
  (err 0.07), Pcc E=5718.51 -> +0.51 above threshold = UNBOUND (method's own 100k eval selection
  had claimed -0.46..-0.83: selection bias). Params counted 1920 (tetra), 2948 (penta).
- Train Pcc failed layout gate on 2 samples at 3e-6 GeV (fp noise) -> tolerance now 1e-4/sample +
  1e-6 GeV mean. Train Tcc R-hat 1.167 -> THIN 20, burn-in +3000 (edited mid-job; official scoring
  in this job uses the edited evaluator). NEXT JOB: delete state/eval_old_reference.done and
  noise_old_reference_* to rescore the old reference with the final evaluator.
- If the retrained reference Pcc is also unbound: Pcc reference sits at the bound (0) -> check
  better variant (full schedule); consider paper's settings (Fig. 2 iterations).
