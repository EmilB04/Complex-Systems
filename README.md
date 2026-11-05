<div align="center">

# Epidemic Spread & Prevention

**Network-based epidemic modelling and evolutionary optimisation of vaccination rules**

**ITI41225 · Complex Systems Modelling and Digital Evolution**  
Østfold University College · Group project · Autumn 2026

**Emil · Andreas · Khalid**

![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)
![Model](https://img.shields.io/badge/Model-Network%20SIR-0F766E)
![Optimisation](https://img.shields.io/badge/Optimisation-Genetic%20Algorithm-7C3AED)
![Status](https://img.shields.io/badge/Status-Experiments%20done%2C%20report%20in%20progress-F59E0B)

</div>

> **Research question**  
> Can an evolved immunisation rule reduce the final size of an epidemic on a contact network, compared with random immunisation, when both rules immunise the same number of individuals before the outbreak starts?

## Overview

Each person is a node in a contact network, and an epidemic spreads through it as a discrete-time **SIR** process. Before the outbreak, a fixed number of people (the budget) are immunised. We compare three ways of choosing them:

| Policy | How people are chosen | Role |
|---|---|---|
| **Random** | Uniformly at random, a new draw for every scenario | Primary baseline |
| **Highest degree** | The people with most contacts | Network-aware reference |
| **Evolved** | Ranked by a weighted sum of three network features; a genetic algorithm (GA) evolves the weights | Proposed method |

As a control, **random search** over the same weights, with the same number of evaluations as the GA, shows whether the GA operators themselves matter.

The epidemic is hypothetical. This is a controlled modelling study, **not** a vaccination recommendation.

## Main result

400 held-out scenarios (20 new networks × 20 outbreaks), budget 20 of 200, β = 0.08, γ = 0.10. Confidence intervals resample whole test networks (cluster bootstrap).

| Policy | Mean infected [95 % CI] | Median | Large outbreaks (>20) |
|---|---:|---:|---:|
| No vaccination | 152.0 [146.4, 157.4] | 178 | 84 % |
| Random | 122.9 [117.3, 128.6] | 152 | 80 % |
| Highest degree | 26.3 [21.3, 31.3] | 4 | 35 % |
| Evolved (GA) | 26.6 [22.6, 30.7] | 4 | 32 % |
| Random search | 26.1 [21.9, 30.4] | 4 | 31 % |

- The evolved rule clearly beats random immunisation: 96.3 fewer infected per network [89.1, 103.3], better on 20 of 20 test networks.
- It does **not** beat highest-degree-first (difference −0.3 [−4.0, 3.4], better on 9 of 20 networks).
- Random search over the same weights is as good as the GA (−0.5 [−1.3, 0.2]).
- All runs converge to a similar rule: large positive weight on degree, negative weight on clustering.

Details, decisions and the full experiment log: [`results/README.md`](results/README.md).

## How the model works

### Contact networks (`network.py`)

Barabási–Albert graphs with n = 200 and m = 3 (591 edges, mean degree 5.91), generated with NetworkX. Every network is static during an outbreak. Separate network seeds are used for training (1–5), testing (301–320, outbreaks 2000–2019) and the pilot (200–209). Networks 101–120 were used for development checks in `test_run.py` and are not part of the final test.

### SIR dynamics (`epidemic.py`)

- **S** susceptible, **I** infectious, **R** recovered or immunised.
- Each step, every infectious person infects each susceptible neighbour with probability β and recovers with probability γ. Updates are **synchronous**: everything is computed from the state at the start of the step.
- Immunised people start in R, are complete and permanent, and are never counted as infected.
- **Patient zero** is drawn among unimmunised people only. The draw depends on the seed alone, so every policy meets the same patient zero whenever that person is unimmunised under both. Comparisons are therefore paired.
- Patient zero and the epidemic use separate random streams (`numpy.random.SeedSequence`).
- `max_steps = 1000`. No run in any experiment reached it.

The simulator returns:

| Output | Meaning |
|---|---|
| `total_infected` | People infected at least once, including patient zero |
| `peak_infected` | Highest number infectious at the same time |
| `duration` | Time steps until no one is infectious |
| `vaccinated_count` | Number of immunised people |
| `initial_infected` | Patient zero |
| `epidemic_ended` | Whether no infectious people remain |
| `history` | S/I/R counts per time step |

### Immunisation policies (`policies.py`)

`random_policy`, `degree_policy` and `weighted_policy`. The weighted policy computes, for each person, normalised degree, normalised mean neighbour degree and the clustering coefficient, scores them as w₁·degree + w₂·neighbour degree + w₃·clustering, and immunises the `budget` highest scores. Weights `(1, 0, 0)` reproduce highest-degree-first.

### Genetic algorithm (`evolution.py`)

| Setting | Value |
|---|---|
| Genome | 3 real-valued weights in [−1, 1] |
| Population | 20 |
| Generations | 20 |
| Selection | Tournament, size 3 |
| Crossover | Uniform (each weight from either parent) |
| Mutation | Gaussian, σ = 0.1, probability 0.2 per offspring, clipped to [−1, 1] |
| Elitism | Best-so-far candidate kept |
| Fitness | Mean infected over 5 training networks × 10 outbreaks (lower is better) |

`evaluate_fitness()` and `run_evolution()` accept one network or a list of networks.

## Getting started

Requires Python 3.14 (versions pinned in `requirements.txt`).

**Windows PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

In VS Code, select the interpreter inside this project's `.venv`, not one from another project.

### Reproduce everything

```bash
python -m pytest -q            # tests
python pilot.py                # pilot: choose beta and budget (~10 s)
python experiments.py --quick  # small check that the pipeline runs (~2 s)
python experiments.py          # main experiments E1 and E2 (~5-8 min)
python analysis.py             # statistics, table and figures (~15 s)
```

`experiments.py` overwrites the CSV files in `results/`. Seeds are fixed, so a rerun on the same commit gives identical files.

Smaller demonstrations: `python main.py` (one outbreak per fixed policy) and `python test_run.py` (one GA run, checked on development networks 101–120).

## Repository structure

```text
Complex-Systems/
├── README.md
├── requirements.txt
├── network.py          # Barabási-Albert contact networks
├── policies.py         # Random, degree and weighted policies, features
├── epidemic.py         # Discrete-time SIR simulation
├── evolution.py        # Genetic algorithm and fitness
├── pilot.py            # Pilot sweep over beta and budget
├── experiments.py      # E1 (main comparison) and E2 (random search)
├── analysis.py         # Bootstrap statistics, LaTeX table and figures
├── main.py             # Quick demo: one outbreak per fixed policy
├── test_run.py         # Quick demo: one GA run, development check
├── results/
│   ├── README.md       # Experiment log and decisions
│   ├── pilot.csv
│   ├── e1_test.csv     # One row per policy x test scenario
│   ├── e1_rules.csv    # Final weights per search run
│   ├── e1_history.csv  # Fitness per GA generation
│   ├── summary.csv     # Table 4
│   ├── paired.csv      # Paired differences with 95 % CI
│   └── table4.tex      # LaTeX rows for the report
├── figures/            # fig_outcomes, fig_fitness, fig_weights, fig_curves (pdf + png)
└── tests/
    ├── test_epidemic.py
    ├── test_evolution.py
    ├── test_integration.py
    ├── test_network.py
    └── test_policies.py
```

## Fair-comparison design

- **Equal budget:** every policy immunises exactly the same number of people.
- **Same scenarios:** all policies meet the same networks, outbreak seeds and disease parameters.
- **Train/test separation:** search uses training networks only; the final test networks (301–320) were never used during development or search.
- **Repeated runs:** 10 independent GA runs and 10 random searches.
- **Equal search effort:** random search gets the same number of evaluations as the GA.
- **Limitations:** perfect immunity, static contacts, a single synthetic network family and a hypothetical disease restrict interpretation.

## Team

| Member | Main implementation responsibility | Shared work |
|---|---|---|
| **Emil** | SIR simulator (`epidemic.py`) and genetic algorithm (`evolution.py`) | Integration, tests, methodological review |
| **Andreas** | Contact networks (`network.py`) and policies (`policies.py`) | Network analysis, baseline review, documentation |
| **Khalid** | Experiment design and pipeline (`pilot.py`, `experiments.py`, `analysis.py`), model fixes before the experiments, statistics and figures | Reproducibility, tests, result presentation |

All members contribute to testing, review and the report.

## Deliverables

Code and a PDF report of about 10–15 pages, together 50 % of the course grade. Submission deadline: **7 December 2026**.

## References and resources

- [NetworkX: Barabási–Albert generator](https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.barabasi_albert_graph.html)
- [NumPy random generation](https://numpy.org/doc/stable/reference/random/index.html)
- [Matplotlib](https://matplotlib.org/stable/)
- [pytest](https://docs.pytest.org/en/stable/)
- [PyCX](https://github.com/hsayama/PyCX) (course resource; not used in the implementation)

External implementations, if reused, are cited and clearly distinguished from the team's own work.

---

<div align="center">
<sub>Complex Systems · Epidemic Spread & Prevention · Østfold University College · 2026</sub>
</div>