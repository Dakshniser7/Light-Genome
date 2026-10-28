# lightgenome: channel-model simulation of transposon-based activity recording

Task simulated: decode **which of K = 4 optogenetic stimulation protocols** (doses 0, 1, 2, 3 by default)
was delivered, from the insertion data of N cells. Information is in bits; the ceiling is log2 4 = 2 bits.

## Files
| file | purpose |
|---|---|
| `model.py` | the 4-layer channel (stimulus → encoding → persistence → readout), mechanistic simulator, exact likelihood tables, Bayes decoders, Monte Carlo mutual information |
| `infotheory.py` | exact MI, Blahut–Arimoto capacity, plug-in / Miller–Madow / shuffle-corrected sample estimators |
| `multiplex.py` | two-colour, two-site-set recorder (site-resolved vs pooled readout) |
| `validate.py` | 27 checks against analytic or brute-force answers (`python validate.py`) |
| `experiments.py` | experiments E1–E6, one JSON each in `results/` (`python experiments.py [e1 ...]`) |
| `figures.py`, `summarize.py` | `figs/*.png`, `results/summary.json` |

Install: `pip install -r requirements.txt` (Python 3.9+). Run everything: `python run_all.py` (or one stage: `python run_all.py --only validate`).
Manual order: `python validate.py && python experiments.py && python figures.py && python summarize.py`
(about 10 minutes on 2 cores; all seeds are fixed, and a full rerun reproduced every number exactly).

## Model in one paragraph
Protocol k has dose S_k. A cell's effective dose is x = S_k·ε_o·z (ε_o: per-cell opsin gain; z: shared per-animal gain).
Transposase activity is h(x) = x/(x+ξ) (saturating) or 2u/(1+u²), u = x/ξ (biphasic, i.e. overproduction inhibition).
Insertions are Poisson with rate λ = b·λ_bg + λ_act·ε_t·h(x) (ε_t: per-cell transposase gain; b: shared baseline scale).
Clonal insertions (mean μ_c, identical within a clone) are added; detection thins each insertion with probability p_d.
`obs="binomial"` replaces the Poisson count by M sites, each hit at most once (CAST-like). All gains are mean-one lognormal.
Per-cell likelihoods are marginalised by quadrature, so population MI is computed with the *exact* Bayes posterior
(Monte Carlo over data only, standard errors reported). No plug-in bias enters any reported number.

## Validation (27/27 pass)
Likelihood rows sum to 1; the mechanistic sampler reproduces the likelihood tables; Poisson thinning gives mean p_d·λ and Fano = 1;
MI = 0 with no coupling and → log2 K for a deterministic channel; Blahut–Arimoto matches the binary-symmetric and Z-channel
closed forms; MC MI at N=1 matches exact MI; total count is sufficient when there is no extrinsic noise (population MI equals the MI of
Poisson(N·p_d·λ)); the clone-aware likelihood matches a brute-force sum; shared-gain marginalisation matches the equivalent single-cell
channel; the shared-nuisance integration is grid-converged; plug-in MI is biased upward and Miller–Madow / shuffle correction reduce it.

Bugs the suite caught during the build (all fixed): the simulator returned one cell instead of N when both per-cell noise terms were zero;
the Miller–Madow correction had the wrong sign; subnormal probabilities produced `inf` MI; and a coarse 7-node quadrature over the shared
nuisance gave nonsense (−430 bits) at N = 10⁴, so the integration now uses a dense grid whose spacing scales as 1/√N, and a guard raises an
error if the exact decoder ever reports negative information.

## Headline results (bits; MC standard errors are about 0.03–0.06 at 300 trials)
**RESULT 1 – endogenous L1 (0.6–13.7 insertions/neuron, ideal detection, no shared nuisance).**
Single-cell capacity over a continuous dose range is 0.0005–0.45 bits. The best corner of the sweep (13.7 per neuron, stimulus-driven
gain equal to baseline) reaches 0.45 bits; at 0.6 per neuron with the same gain it is 0.04 bits. Resolving four protocols to 90 % of 2 bits takes
275–2,566 independent neurons when the gain is 1, and 1,700 to more than 10⁴ when it is 0.25. So single-neuron L1 recording is far below
1 bit, and what matters is the number of cells, not the rate.

**RESULT 2 – engineered recorders (p_d = 0.3 assumed).**
Calling Cards-like: capacity 0.08 bits at 1 peak insertion per cell, 0.41 at 5.6, 0.92 at 31.6; 90 % of 2 bits needs about 2,200 / 910 / 350 cells at
peak 1 / 3 / 10. CAST-like (10–25 % per-site integration): a single site needs more than 3,000 cells; 14 sites at 25 % needs about 720.
Extrinsic expression noise flattens the capacity gain: at 300 peak insertions the capacity is 2.30 bits with no noise but 1.82, 1.44 and 1.02 bits
at log-sd 0.25, 0.5 and 1 (E6).

**RESULT 3 – clones, inhibition, shared nuisance.**
* *Clonality (E3a).* A decoder that ignores clonal sharing collapses (down to negative bits, i.e. worse than the prior, because the posterior is
  overconfident). A clone-aware decoder that knows clone membership recovers essentially the no-clone value once clones are large enough
  (≥5–10 cells), because the shared offset can be read from the lowest-count cells. The benefit depends on membership being known; with
  count-only data and unknown membership it is not guaranteed.
* *Overproduction inhibition (E3b).* With a biphasic response, information is non-monotone in stimulus intensity: at N = 1 it peaks near S_max ≈ 2
  (0.18 bits) and falls to 0.03 bits at S_max ≈ 40; at N = 30 it falls from 1.27 to 0.57 bits between S_max ≈ 9 and 40. There is also an aliasing dip
  when the protocol doses straddle the activity peak (S_max ≈ 2–3 at N = 30). With a saturating response nothing comparable happens.
* *Shared per-animal nuisance (E4).* An unknown shared gain z creates an N-independent ceiling: at N = 1,000, MI is 1.82 bits with no nuisance,
  1.49 at σ_z = 0.25 and 1.14 at σ_z = 0.5. For L1, an unknown per-animal baseline (σ_b = 0.35) drops MI at N = 1,000 from 1.72 to 0.40 bits.
  MI is not monotone in σ_b (0.40, 0.48, 0.77 for σ_b = 0.35, 0.7, 1.4) because the mean-one lognormal puts more animals at low baseline, where
  decoding is easy; this is exact marginalisation, not an artefact, but it is a property of the chosen parametrisation.

**RESULT 4 – multiplexing (E5).** With two colours driving two site-sets and no cross-talk, a site-resolved readout reaches 2.0 bits at N = 30 while a pooled
count tops out at 1.5 bits (outcomes ¼, ½, ¼). Cross-talk χ = 0.3 lowers the resolved readout to 1.86 bits at N = 100, and at χ = 0.6 it falls to 1.33.

## Assumptions and limits
1. **p_d = 0.3, the Calling Cards peak rates, and all noise levels are assumed values**, not taken from Moudgil et al. (2020) or any other cited paper. Measured values should replace them when available.
2. **The L1 activity gain r is a free parameter.** Muotri et al. (2010) and Bedrosian et al. (2018) are suggestive only; there is no measured transfer function from firing to insertion rate. The grid (r = 0.1–1) is a sweep, not a finding.
3. L1 baseline is treated as private noise, except in E3a where a fraction φ is clonal. That moves background from private to shared, so the L1 clonal panel conflates two effects.
4. Cells are conditionally independent given protocol, apart from the explicit clonal and shared terms. Readout is a **count per cell** (site identity only in E5).
5. CAST-like clones are not modelled; E3a covers Poisson recorders only.
6. **No kinetics.** Transposition is treated as integrating over the whole stimulation window. Any statement about the timescale needs a cited source.
(The old line mentioned the abstract, which no longer applies to this wording.)
7. The biphasic response and its ξ are hypothetical (motivated by overproduction inhibition reported for Sleeping Beauty by Wu et al., 2006).
8. Shared-nuisance runs are limited to N ≤ 1,000 and one random shared variable at a time; larger N would need a finer grid.
9. N-for-90 % values are log-interpolated from a coarse N grid with Monte Carlo noise, so treat them as ±20–30 %.
10. Capacities are over a dose range of 0–3 in units of ξ with the stated noise. Different ranges give different numbers.
