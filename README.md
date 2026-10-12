# Epidemic Spread and Prevention

Group project for **ITI42622 — Complex Systems Modelling and Optimization** at **Østfold University College**.

**Project theme:** Alternative 1 — Epidemic spread and prevention  
**Team:** Emil, Andreas and Khalid  
**Submission deadline:** 7 December 2026  
**Status:** Project setup and planning. Implementation and experiments are pending.

## Project overview

This project will investigate whether an evolved prevention strategy can reduce epidemic spread more effectively than a simple fixed strategy under the same intervention budget.

The planned approach combines a network-based epidemic model with an evolutionary algorithm. Individuals are represented as nodes in a contact network, and connections represent opportunities for transmission. Before an outbreak begins, a limited number of individuals can be immunised. The evolutionary algorithm will search for a rule that determines which individuals to prioritise.

The epidemic is hypothetical. The model is intended to study local interactions, collective outbreak behaviour and prevention strategies under controlled assumptions.

## Research question

> Can an evolved prioritisation rule reduce the average final epidemic size compared with random immunisation when both strategies immunise the same number of individuals?

The main modification will be the allocation of immunisation. The epidemic model and intervention budget will remain the same in the primary comparison.

## Planned model and methods

The following choices are provisional and will be confirmed through implementation and pilot experiments.

### Epidemic model

We plan to use a discrete-time, network-based **SIR model**:

- **Susceptible (S):** individuals who can become infected.
- **Infected (I):** individuals who can transmit the infection to their contacts.
- **Recovered/Removed (R):** individuals who no longer participate in transmission.

The initial model will use a static contact network, synchronous updates and immunisation before the outbreak. Immunised individuals will be tracked separately from individuals who become infected and subsequently recover.

### Prevention strategies

| Strategy | Planned role |
|---|---|
| Random immunisation | Main baseline: select individuals uniformly at random without replacement. |
| Evolved prioritisation | Proposed strategy: rank individuals using an evolved combination of network features. |
| Highest-degree-first | Additional reference: prioritise individuals with the most contacts. |
| No immunisation | Optional descriptive reference with a different intervention budget. |

All strategies in the budget-matched comparison will immunise exactly the same number of individuals.

### Evolutionary optimisation

The proposed policy representation is a small real-valued weight vector combining network features such as degree, average neighbour degree and clustering coefficient.

A **(μ + λ) evolution strategy** is the current recommendation. Candidate policies will be evaluated through repeated epidemic simulations, varied through Gaussian mutation and selected according to fitness. Fitness will reward a lower average final epidemic size.

Population sizes, mutation settings, network parameters and simulation counts will be recorded once the experimental design is finalised.

## Experimental plan

The planned evaluation will:

1. Verify the simulator on small networks with predictable outcomes.
2. Run pilot experiments to check model behaviour and estimate computational cost.
3. Separate training scenarios, validation scenarios and a held-out final test set.
4. Repeat evolutionary searches and epidemic simulations to show variation.
5. Compare strategies under the same intervention budget and documented test conditions.
6. Report quantitative effects, uncertainty, computational cost and model limitations.

The primary outcome will be the fraction of the total population that becomes infected at any point during the outbreak, including initial infections. Immunised individuals will not be counted as infected. Secondary outcomes may include peak prevalence and outbreak duration.

Negative or inconclusive results will be reported and discussed.

## Planned tools

| Tool | Purpose |
|---|---|
| Python | Simulation, optimisation and experiment automation. |
| NumPy | Numerical operations and controlled random-number generation. |
| NetworkX | Contact networks and network features. |
| Matplotlib | Figures and visual analysis. |
| pytest | Automated tests. |
| Overleaf / LaTeX | Report writing and PDF generation. |

PyCX may be used as a learning resource or implementation starting point. Any reused code will be attributed, and modifications will be documented. DEAP may be considered if it helps implement and explain the evolutionary algorithm.

## Proposed repository structure

This is a planned layout; the directories and files may not exist yet.

```text
.
├── README.md
├── requirements.txt
├── src/
│   └── epidemic_project/
│       ├── networks.py
│       ├── simulator.py
│       ├── policies.py
│       ├── evolution.py
│       ├── experiments.py
│       └── analysis.py
├── configs/
├── seed_manifests/
├── tests/
├── results/
├── figures/
└── report/
    ├── main.tex
    ├── references.bib
    └── sections/
        ├── 01_introduction.tex
        ├── 02_model.tex
        ├── 03_experiments.tex
        ├── 04_results.tex
        ├── 05_discussion.tex
        ├── 06_conclusion.tex
        └── appendix.tex
```

The report is being prepared in Overleaf. The `report/` directory is intended for a versioned copy of its source files.

## Reproducibility and verification

Installation and execution instructions will be added when the corresponding code is available. The completed repository should include:

- The Python version and pinned dependency versions.
- Commands to run tests, experiments and figure generation.
- Experiment configurations and seed manifests.
- Selected policy weights and recorded simulation outcomes.
- The code version associated with the submitted report.

Planned tests include no-transmission cases, isolated networks, synchronous update behaviour, immunity handling, exact intervention budgets, reproducibility and a small end-to-end experiment.

## Collaboration

Initial coordination responsibilities are:

| Team member | Main areas |
|---|---|
| Emil | Epidemic simulator, random-seed organisation and evolutionary loop. |
| Andreas | Networks, feature extraction, prevention policies and fitness integration. |
| Khalid | Experiment automation, analysis and test infrastructure. |

All three members will contribute to code, testing, report writing and review. Responsibilities will be adjusted to keep the workload balanced, and every member should understand the complete solution.

## Deliverables

- A PDF report containing the project description and research question, model and evolutionary method, experiments, results and discussion.
- Code provided through a repository link in the report or as a compressed ZIP archive.
- Instructions and recorded configurations for reproducing the reported experiments.

## Resources

These are starting resources. Sources actually used in the implementation and report will be cited explicitly.

- [PyCX simulation examples](https://github.com/hsayama/PyCX)
- [Hiroki Sayama: Introduction to the Modeling and Analysis of Complex Systems](https://bingdev.binghamton.edu/sayama/textbook/)
- [NetworkX documentation](https://networkx.org/documentation/stable/)
- [NumPy random sampling documentation](https://numpy.org/doc/stable/reference/random/index.html)
- [DEAP evolutionary algorithms](https://deap.readthedocs.io/en/master/api/algo.html)
- [pytest documentation](https://docs.pytest.org/en/stable/)
