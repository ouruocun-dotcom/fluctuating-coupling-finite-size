# Fluctuating coupling and finite-population synchronization
Reproduction materials for *Fluctuating coupling mimics synchronization in finite oscillator populations*, Ruocun Ou.

## Contents
- README.md: installation, commands, data definitions and scope.
- requirements.txt: Python dependencies.
- micro.py, special.py, controls.py: microscopic white/colored integrators, prescribed-drift and colored mesoscopic algorithms, and paired/critical controls.
- simulate.py: regenerate a selected condition using its recorded seed.
- reproduce.py: recalculate statistics and compact review plots.
- conditions.json: parameters and seeds for all 142 saved conditions.
- replica_statistics.npz: compressed numerical data for 4872 replicas.

No manuscript, correspondence, internal records or audit reports are included.

## Installation and quick reproduction
Use Python 3.12:
```bash
python -m pip install -r requirements.txt
python reproduce.py --output results
```
Run this from any directory; inputs are resolved relative to the script.
It writes moments.csv, exponents.csv, paired_steps.csv, blocks.csv, tails.csv, and six review figures. These compact plots reproduce the observables, not the exact publication styling. Figures 3, 4 and 6 emphasize the saved microscopic points; white-density predictions are also directly callable as density()/theory(). Figure 5 contains closure curves. Statistics are computed from the included data, not pasted from manuscript tables.

## Regenerating a time series
Look up an id in conditions.json, then:
```bash
python simulate.py --id c000 --replicas 1 --output rerun.npy
```
Omit --replicas to run the saved replica count. Existing output files are not overwritten.
Full runs, especially large populations, can take hours. This is a selected-condition runner, not a short demonstration of scientific convergence.

The data-generation environment used NumPy 2.5.4 and Numba 0.68.0. Matching random-number seeds across different library versions need not produce bitwise-identical trajectories. Included saved statistics do not require Numba to analyze; Numba is required to regenerate trajectories. Simulation functions preserve the original random-draw order. Computational smoke checks are not new scientific data.

## Data definitions and scope
This minimal package stores per-replica statistics, not complete time series. Original time-series arrays remain available from the author upon reasonable request; simulate.py regenerates conditions. The compressed data permit moment, tail-occupation, six-block and paired-step reanalysis, but cannot support arbitrary new time-lag or threshold analyses.

Keys in replica_statistics.npz:
- thresholds: radii at which time fractions were measured (the manuscript grid plus 0.1, 0.2, 0.3 and 0.4).
- cXXX_moments: (replica, branch, 3), moments of R of orders 1, 2 and 4.
- cXXX_blocks: (replica, branch, 6), mean R squared in six equal time blocks.
- cXXX_tail: (replica, branch, threshold), time fraction with R greater than threshold.

Original float32 R-squared samples were promoted to float64 before reducing. Ordinary conditions have one branch. Paired conditions have four ordered branches: original coarse, original fine, split coarse, split fine. Branches within a replica share the same frequency/initial-phase sample and nested Brownian increments; bootstrap whole pairs, not branches independently.

Each condition provides N, K, D, tau, scheme, dt, burn, T, M, seed0, and kind. scheme 0 is white stochastic Heun, 1 white Euler-Maruyama, 2 colored microscopic coupling. A tau of 1 in white configurations is an unused placeholder. kind identifies micro, meso, correction_a, correction_b, rk4, shared, paired or critical. Seeds are seed0 plus the zero-based replica index. Shared-frequency groups of eight use the special.py frequency-seed rule. The initial batch includes base, extra-replica and step variants; base/extra are pooled where indicated, step variants are not pooled with production results.

Initial-scan conditions can share random streams across K or memory; they are not independent cross-condition confirmations. Independent repeats use disjoint condition seeds. Drift variants are deliberately paired. The shared-frequency condition contains five independent frequency groups, not forty independent groups. Its moment SE is computed over group means; its Binder interval is not used as scientific evidence.

Means and uncertainty use whole replicas. Scaling fits are unweighted log-log regressions, with 2000 bootstrap draws within each population size; Binder and paired intervals use 10000 draws. Bootstrap intervals can differ slightly from published values because of Monte Carlo resampling order, while point estimates agree. Shared-frequency group dependence and original cross-condition stream sharing must not be ignored.

Kesten tails and moment laws are closure asymptotics; curved finite-size tails are not direct evidence of a microscopic pure power. Negative Binder values are not a unique noise identifier. The package retains discrepant controls as well as successful comparisons. Fixed-grid quadrature and tested numerical step differences are not universal convergence guarantees.

