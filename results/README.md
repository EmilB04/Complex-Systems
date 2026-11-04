## E1 and E2, final run on unseen test seeds

- **Why:** `test_run.py` evaluated intermediate rules on networks 101-120 with outbreaks 1000-1009 while the patient-zero defect was being fixed. The first main run (commit fb1e73a, above) used 101-120 / 1000-1019 as its test set, so that set had influenced the corrections. The experiment was repeated on fresh seeds. Networks 101-120 are now the development set.
- **Change:** `test_graph_seeds` 101-120 -> 301-320, `test_outbreak_seeds` 1000-1019 -> 2000-2019. Training, search and all settings unchanged, so the 20 rules are identical to the first run.
- **Code version:** commit c3ccd07 (branch feat/experiments)
- **Date:** 2026-11-04
- **Hardware:** Windows 11 Home, Intel Core i7-10750H @ 2.60 GHz, Python 3.14.4, NetworkX 3.7, NumPy 2.5.4
- **Runtime:** 446 s. A second machine gave identical output files.
- **Truncated runs:** 0

| Policy | Mean infected [95 % CI] | Median | Large (>20) | Mean peak |
|---|---:|---:|---:|---:|
| none | 152.0 [146.4, 157.4] | 178 | 84 % | 72.0 |
| random | 122.9 [117.3, 128.6] | 152 | 80 % | 55.4 |
| degree | 26.3 [21.3, 31.3] | 4 | 35 % | 7.8 |
| evolved | 26.6 [22.6, 30.7] | 3.5 | 32 % | 7.9 |
| random search | 26.1 [21.9, 30.4] | 4 | 31 % | 7.8 |

Paired differences per network (positive = fewer infections for the second policy):

- random - evolved: 96.3 [89.1, 103.3], second better on 20/20
- degree - evolved: -0.3 [-4.0, 3.4], 9/20
- random_search - evolved: -0.5 [-1.3, 0.2], 10/20
- random - degree: 96.6 [89.4, 104.0], 20/20

Per-run test mean: evolved 24.5-33.5 (sd 2.8), random search 24.4-35.1 (sd 3.3). `fig_curves` uses evolved run 10 (median run).

**Conclusions unchanged from the first run.** These are the numbers reported in the paper.