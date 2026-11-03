# Experiment log

All numbers in the report are computed from the CSV files in this folder.
Seeds and code are committed, so every file can be regenerated.

## Development notes (test_run.py, one network, before the changes)

These runs are not results. They explain why the model was changed
before the main experiments.

- Original setup (3 training outbreaks, patient zero drawn from all 200,
  weights in [0, 1], max_steps 100): training fitness 0.67, test fitness
  75.6, worse than random (about 66). The GA reached 0.67 in generation 1
  by vaccinating patient zero in the training outbreaks, and never improved.
- Patient zero drawn only among unvaccinated people: training 1.0, test 34.4.
- max_steps 100 -> 1000: about 10 % of unvaccinated outbreaks lasted longer
  than 100 steps (longest 130).
- Weights in [-1, 1]: the GA chose a negative weight for the first time
  (test 24.7, still only 3 training outbreaks, mostly noise).
- Separate random streams for exposure, epidemic and GA: test 3.8
  (still noise, 3 training outbreaks).
- 5 training networks x 10 outbreaks (50 scenarios), weights in [-1, 1],
  beta 0.05: training 5.8, test 6.5, degree 6.1. Overfitting gone, but
  degree already stopped almost all outbreaks at beta 0.05.

## Pilot (pilot.py)

- Purpose: choose beta and budget before the main experiments
- Networks: Barabási-Albert, n=200, m=3, seeds 200-209
- Outbreaks: seeds 5000-5019 (4200 simulations in total)
- Policies: none, random, degree
- Grid: beta in {0.03, 0.05, 0.08}, budget in {10, 20, 30}
- Output: pilot.csv. No truncated runs.
- Decision: beta=0.08, budget=20. Random gives 110.8 infected on average,
  degree 24.5, and degree still has 32 % large outbreaks. At beta=0.05,
  budget=20 degree already stops almost all outbreaks (6.0 infected on
  average), leaving nothing for an evolved rule to improve.
- Alternative setting for a robustness check: beta=0.05, budget=10
  (random 72.8, degree 24.1).

## E1 and E2 (experiments.py)

- Command: `python experiments.py`
- Code version: commit fb1e73a (branch feat/experiments)
- Date: 2026-11-03
- Hardware: Windows 11 Home, Intel Core i7-10750H @ 2.60 GHz
- Software: Python 3.14.4, NetworkX 3.7, NumPy 2.5.4
- Runtime: 452 s
- Model: beta=0.08, gamma=0.10, budget=20, n=200, m=3, max_steps=1000
- Training: networks 1-5 x outbreaks 10-19 (50 scenarios)
- Test: networks 101-120 x outbreaks 1000-1019 (400 scenarios),
  never used during search
- GA: 10 independent runs (seeds 1-10), population 20, 20 generations,
  mutation rate 0.2, sigma 0.1, tournament size 3
- Random search (E2): same number of candidate evaluations as the GA run
  with the same seed (118-189 per run)
- Output: e1_test.csv, e1_rules.csv, e1_history.csv
- Truncated runs: 0

| policy        | mean infected | median | large outbreaks (>20) |
|---------------|--------------:|-------:|----------------------:|
| none          | 141.4 | 179.0 | 78 % |
| random        | 117.1 | 151.0 | 76 % |
| degree        |  31.0 |   4.0 | 36 % |
| evolved       |  31.5 |   5.0 | 38 % |
| random search |  30.6 |   5.0 | 37 % |

- Evolved: mean per run 29.4-35.0 (sd 1.5)
- Random search: mean per run 28.8-37.5 (sd 2.6)
- Training fitness: evolved 23.1-26.2, degree 31.8 on the same training
  scenarios. The advantage over degree did not carry over to the test networks.
- All 10 GA runs ended with a positive degree weight and a negative
  clustering weight.