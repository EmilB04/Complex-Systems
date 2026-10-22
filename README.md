<div align="center">

# Epidemic Spread & Prevention

**Network-based epidemic modelling and evolutionary optimisation of vaccination strategies**

**ITI42622 · Complex Systems Modelling and Optimization**  
Østfold University College · Group project · Autumn 2026

**Emil · Andreas · Khalid**

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![Model](https://img.shields.io/badge/Model-Network%20SIR-0F766E)
![Optimisation](https://img.shields.io/badge/Optimisation-Genetic%20Algorithm-7C3AED)
![Status](https://img.shields.io/badge/Status-Experiments%20in%20progress-F59E0B)

</div>

> **Research question**  
> Can an evolved prevention strategy reduce epidemic spread more effectively than a simple fixed strategy under the same intervention budget?

## Overview

This project investigates how local disease transmission on a social-contact network produces population-level outbreaks, and whether **evolutionary search** can improve the allocation of a limited vaccination budget.

Each person is represented by a node, each edge represents a possible contact, and disease states evolve according to a discrete-time **SIR (Susceptible–Infected–Recovered)** model. A genetic algorithm searches over sets of people to vaccinate **before** an outbreak. Its performance will be evaluated against **random vaccination** (the primary baseline) and **highest-degree-first vaccination** (an additional fixed reference strategy) using the **same number of vaccines**.

The epidemic is hypothetical. This is a controlled modelling and optimisation study, **not a real-world vaccination recommendation or epidemiological forecast**.

### Project status

| Component | Status | Notes |
|---|---|---|
| Contact-network generation | ✅ Implemented | Barabási–Albert network using NetworkX |
| Fixed prevention policies | ✅ Implemented | Random and highest-degree-first vaccination |
| Network feature extraction | ✅ Implemented | Degree, mean neighbour degree and clustering coefficient; not used by the current GA |
| Epidemic simulation | ✅ Implemented | SIR dynamics, vaccination, synchronous updates, outcome history |
| Genetic algorithm | ✅ Implemented | Initial population, fitness, tournament selection, crossover, mutation and elitism |
| Training/held-out evaluation | ✅ Prototype | Separate random seeds; full statistical comparison pending |
| Automated tests | ✅ Implemented | Run the repository tests to check your checkout |
| Repeated comparative experiments and plots | 🚧 In progress | `experiments.py` is a placeholder in this snapshot |
| Final report and interpretation | 📋 Planned/in progress | Report prepared separately in Overleaf |

## How the model works

### 1. Contact network

`network.py` creates a **Barabási–Albert (BA)** graph. The model produces a heterogeneous contact structure: some nodes have substantially more connections than others. The graph is **static** during an epidemic run.

Current demonstration settings:

| Parameter | Example | Meaning |
|---|---:|---|
| `n` | `200` | Number of individuals |
| `m` | `3` | Number of attachments for each new node in the BA generator |
| `seed` | `42` | Reproducible network generation |

The network is a simplified representation of possible contacts, not observed human contact data.

### 2. SIR epidemic dynamics

`epidemic.py` simulates each time step using the states:

- **S — Susceptible:** not immune and not yet infected.
- **I — Infected:** currently infectious and able to transmit to susceptible neighbours.
- **R — Recovered/immune:** recovered after infection or immunised before the outbreak.

For each infectious–susceptible contact, transmission occurs with probability `beta`. Each currently infected person recovers with probability `gamma` per step. State changes are applied **synchronously**, so a newly infected individual cannot transmit until the next step. Vaccination is modelled as **complete, immediate immunity** and consumes one unit of the intervention budget per person.

An initial exposure is selected independently of vaccination status. If that person has been vaccinated, the exposure does **not** establish an infection. The model therefore allows vaccination to prevent the outbreak from starting, an important assumption to account for in evaluations.

Example simulation settings are `beta=0.05`, `gamma=0.10`, and `max_steps=100`. These are demonstration parameters for a hypothetical disease, not estimated medical values.

The simulator returns:

| Output | Meaning |
|---|---|
| `total_infected` | Number of people infected at least once, including initial infections |
| `peak_infected` | Highest simultaneous number of infected individuals |
| `duration` | Simulated time steps elapsed |
| `vaccinated_count` | Number of vaccinated individuals |
| `epidemic_ended` | Whether no infected individuals remain |
| `history` | S/I/R counts by time step |

**Important:** A run that hits `max_steps` with `epidemic_ended=False` may have an incomplete final outbreak size. Such cases need to be checked in the experimental analysis.

### 3. Prevention policies

`policies.py` currently implements two fixed policies:

| Strategy | Selection rule | Role |
|---|---|---|
| **Random vaccination** | Randomly select distinct nodes | Primary baseline |
| **Degree vaccination** | Vaccinate nodes with the highest number of neighbours | Additional, network-informed reference |
| **Evolved vaccination** | Genetic algorithm searches for a set of vaccinated node IDs | Proposed method |

All three budget-matched strategies must select **exactly the same number** of individuals on the same network. The example budget is `20` out of `200` nodes (10%).

### 4. Evolutionary optimisation

`evolution.py` uses a **genetic algorithm (GA)** rather than the weighted-feature evolution strategy considered in the early planning notes.

A candidate solution is a **set of unique node IDs**, with set size equal to the vaccine budget. The algorithm:

1. Creates an initial population of vaccination sets.
2. Evaluates each set using repeated SIR simulations.
3. Assigns **fitness = average number of infected individuals**; lower is better.
4. Selects parents using **tournament selection**.
5. Produces offspring by combining vaccinated nodes from the parents (**crossover**).
6. Optionally swaps one vaccinated node for an unvaccinated node (**mutation**).
7. Preserves the best candidate through **elitism** and repeats over generations.

The population is evaluated using fixed training seeds within a run; repeated identical candidates are cached. The code records best-so-far fitness and population-average fitness per generation.

A typical development configuration is `population_size=20`, `generations=15`, `mutation_rate=0.2`, `tournament_size=3`, and `fitness_seeds=range(100, 130)`. These are **pilot settings**, not final experimentally validated hyperparameters.

> **Interpretation caveat:** Fixed training exposure seeds can allow the GA to fit those particular outbreak starting points. A low *training* fitness does not demonstrate better *general* prevention. Held-out testing and repeated independent runs are therefore essential.

## Getting started

### Prerequisites

- Python **3.13** (used during development; other versions may work but are not currently verified)
- Git and a terminal
- Recommended: VS Code with the Python extension

Clone the repository or open the project directory, then create a virtual environment in the **project root**.

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**Windows PowerShell**

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If using VS Code, select the Python interpreter inside this project's `.venv`. Avoid reusing a virtual environment from another project.

### Run the simulation

```bash
python main.py
```

This runs the existing **random-vs-degree demonstration**. It is a sanity check, **not** a statistically controlled final comparison; the initial exposure must be explicitly controlled for the formal experiments.

### Run the genetic algorithm

```bash
python evolution.py
```

This runs the current development demonstrations for population generation, mutation, fitness, tournament selection and crossover, followed by evolutionary optimisation and a held-out test. Results are printed in the terminal. The demonstration settings currently use 30 training seeds and 100 held-out test seeds in the evolution run.

### Run the tests

```bash
python -m pytest tests/ -v
```

Keep the tests passing as the model and experimental pipeline evolve.

## Repository structure

```text
Complex-Systems/
├── README.md              # Project information and instructions
├── requirements.txt       # Python dependencies
├── main.py                # Random-vs-degree demonstration
├── network.py             # Barabási–Albert contact network
├── policies.py            # Fixed policies and network features
├── epidemic.py            # Discrete-time SIR simulation
├── evolution.py           # Genetic algorithm and demo run
├── experiments.py         # Planned experimental pipeline (placeholder)
└── tests/
    ├── test_epidemic.py
    ├── test_evolution.py
    └── test_policies.py
```

Temporary folders such as `.venv/`, `__pycache__/` and `.pytest_cache/` are local-only and should not be committed. The report is currently prepared separately in Overleaf; the diagram above reflects the Python project snapshot, not a proposed future package layout.

## Experimental design — planned

The central evaluation will answer the research question by comparing **random**, **degree**, and **evolved** vaccination under a common budget.

**Primary outcome:** average final epidemic size (total infected, optionally divided by population size).  
**Secondary outcomes:** peak concurrent infections, outbreak duration, epidemic extinction, and variability across runs.

Proposed experimental workflow:

1. Validate the SIR simulator on small, predictable networks.
2. Fix a network and vaccination budget for the primary controlled comparison.
3. Optimise using **training seeds** only; do not select policies based on the final test results.
4. Evaluate all policies on **identical held-out exposure seeds** and matching disease parameters.
5. Repeat random-policy draws and complete GA optimisation runs to estimate run-to-run variation.
6. Report means and dispersion (e.g., standard deviation or confidence intervals), not only the best run.
7. Check unfinished epidemics (`epidemic_ended=False`) and the effect of the step limit.
8. Where feasible, repeat on different network seeds to test dependence on a single contact graph.
9. Produce figures and tables and discuss cases where the evolved policy **fails** to outperform a fixed baseline.

Potential plots include epidemic S/I/R curves, GA best/mean fitness by generation, and strategy-comparison distributions on held-out runs. The final protocol, sample sizes, saved artefacts and parameter configurations will be documented when `experiments.py` is implemented.

### Fair-comparison considerations

- **Equal budget:** all policies use the same number of vaccines.
- **Same graph and disease parameters:** policy is the primary independent variable.
- **Controlled exposure:** initial exposure must not be resampled as a function of the vaccinated set.
- **Train/test separation:** held-out epidemic seeds must not guide policy selection.
- **Stochastic variation:** repeated seeds and independent optimisation runs are needed.
- **Model limitations:** perfect immunity, static contacts and a hypothetical SIR process restrict interpretation.

## Team and responsibilities

| Member | Main implementation responsibility | Shared work |
|---|---|---|
| **Emil** | SIR simulator (`epidemic.py`) and evolutionary search (`evolution.py`) | Integration, tests, methodological review |
| **Andreas** | Contact networks (`network.py`) and prevention policies (`policies.py`) | Network analysis, baseline review, documentation |
| **Khalid** | Experiments (`experiments.py`), statistics and visualisations | Reproducibility, analysis, result presentation |

All three members are expected to understand the model, contribute to testing and review, and participate in writing the report.

## Report and deliverables

The project is assessed through **both code and a written report** (approximately **10–15 pages**), together worth **50% of the course grade**. The report must be submitted as **PDF**, with code provided either via a repository link included in the report or as a ZIP attachment.

Required report coverage includes:

- Project description and research question
- Model description, including baseline, proposed approach and evolutionary method
- Experiments and evaluation setup
- Results and quantitative comparisons
- Discussion, assumptions and limitations

The model must be understood as a **complex system**, with interacting individuals, local infection/recovery rules, and emergent population-level epidemic behaviour. Evolution must make a clearly explained modification to the prevention policy, and comparisons must respect the same resource budget. Repeated trials and explained negative results are acceptable and important.

**Submission deadline: 7 December 2026.**

## References and learning resources

- **Assignment brief:** *ITI42622 — Complex Systems Modelling and Optimization*, Alternative 1: *Epidemic spread and prevention* (course-provided PDF; authoritative requirements).
- [NetworkX: Barabási–Albert generator](https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.barabasi_albert_graph.html)
- [NetworkX documentation](https://networkx.org/documentation/stable/)
- [NumPy random generation](https://numpy.org/doc/stable/reference/random/index.html)
- [pytest documentation](https://docs.pytest.org/en/stable/)
- [Matplotlib documentation](https://matplotlib.org/stable/)
- [PyCX project](https://github.com/hsayama/PyCX) — optional learning resource suggested by the assignment

External implementations, if reused later, must be cited and clearly distinguished from the team's contribution.

---

<div align="center">
<sub>Complex Systems · Epidemic Spread & Prevention · Østfold University College · 2026</sub>
</div>
