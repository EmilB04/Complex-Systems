# Khalid - Main experiments (E1 and E2)
"""
E1: Evolved rule versus random immunisation, with highest-degree-first
    and no vaccination as references.
E2: Random search over the same weights with the same number of
    candidate evaluations as the GA. Shows whether the GA operators
    matter, or whether evaluating many candidates is enough.

Design
- Training: 5 networks x 10 outbreaks (50 scenarios). Only these are
  used to compute fitness.
- Test: 20 new networks x 20 outbreaks (400 scenarios). Never used
  during search.
- 10 independent GA runs and 10 independent random searches.
- Every policy meets the same test scenarios. A scenario is a
  (network, outbreak seed) pair, and simulate_epidemic picks the same
  patient zero for every policy whenever that person is unvaccinated.

Output (in results/)
- e1_test.csv: one row per policy x test scenario
- e1_rules.csv: weights and training fitness for every search run
- e1_history.csv: best and average fitness per GA generation

Run: python experiments.py           (full run)
     python experiments.py --quick   (small run to check that it works)
"""

import argparse
import csv
import os
import time

import numpy as np

from epidemic import simulate_epidemic
from evolution import evaluate_fitness, run_evolution
from network import create_networks
from policies import degree_policy, random_policy, weighted_policy

# Model settings, chosen in the pilot (results/pilot.csv)
BETA = 0.08
GAMMA = 0.10
BUDGET = 20
N = 200
M = 3

FULL = {
    "train_graph_seeds": range(1, 6),
    "train_outbreak_seeds": range(10, 20),
    "test_graph_seeds": range(301, 321),
    "test_outbreak_seeds": range(2000, 2020),
    "search_seeds": range(1, 11),
    "population_size": 20,
    "generations": 20,
}

QUICK = {
    "train_graph_seeds": range(1, 3),
    "train_outbreak_seeds": range(10, 13),
    "test_graph_seeds": range(101, 104),
    "test_outbreak_seeds": range(1000, 1005),
    "search_seeds": range(1, 3),
    "population_size": 6,
    "generations": 3,
}


def random_search(graphs, seeds, n_candidates, search_seed):
    """
    Evaluate n_candidates random weight vectors and keep the best.

    Uses the same fitness function, training scenarios and number of
    evaluations as the GA, but no selection, crossover or mutation.
    """

    rng = np.random.default_rng(search_seed)
    best_weights, best_fitness = None, float("inf")

    for _ in range(n_candidates):
        weights = rng.uniform(-1, 1, size=3)
        fitness = evaluate_fitness(
            graphs, weights, seeds, BUDGET, beta=BETA, gamma=GAMMA
        )

        if fitness < best_fitness:
            best_weights, best_fitness = weights, fitness

    return best_weights, best_fitness


def run_search(config, train_graphs):
    """Run the GA and the random search once per search seed."""

    rules = []
    history = []

    for search_seed in config["search_seeds"]:
        ga = run_evolution(
            train_graphs,
            BUDGET,
            population_size=config["population_size"],
            generations=config["generations"],
            fitness_seeds=list(config["train_outbreak_seeds"]),
            seed=search_seed,
            beta=BETA,
            gamma=GAMMA,
            verbose=False,
        )

        rules.append({
            "method": "evolved",
            "run": search_seed,
            "weights": ga["best_weights"],
            "train_fitness": ga["best_fitness"],
            "evaluations": ga["evaluations"],
        })

        for row in ga["fitness_history"]:
            history.append({"run": search_seed, **row})

        # Same number of candidate evaluations as this GA run
        rs_weights, rs_fitness = random_search(
            train_graphs,
            list(config["train_outbreak_seeds"]),
            ga["evaluations"],
            search_seed,
        )

        rules.append({
            "method": "random_search",
            "run": search_seed,
            "weights": rs_weights,
            "train_fitness": rs_fitness,
            "evaluations": ga["evaluations"],
        })

        print(
            f"Run {search_seed}: GA train {ga['best_fitness']:.2f} "
            f"| random search train {rs_fitness:.2f} "
            f"| {ga['evaluations']} evaluations each"
        )

    return rules, history


def test_policies(config, rules):
    """Run every policy on every test scenario and return one row each."""

    rows = []

    for graph_seed in config["test_graph_seeds"]:
        G = create_networks([graph_seed], n=N, m=M)[0]

        # Fixed allocations that do not depend on the outbreak seed
        fixed = {
            ("none", None): set(),
            ("degree", None): degree_policy(G, BUDGET),
        }

        for rule in rules:
            key = (rule["method"], rule["run"])
            fixed[key] = weighted_policy(G, BUDGET, rule["weights"])

        for outbreak_seed in config["test_outbreak_seeds"]:
            allocations = dict(fixed)

            # A new random allocation for every scenario
            allocations[("random", None)] = random_policy(
                G, BUDGET, seed=outbreak_seed
            )

            for (policy, run), vaccinated in allocations.items():
                result = simulate_epidemic(
                    G,
                    vaccinated=vaccinated,
                    beta=BETA,
                    gamma=GAMMA,
                    seed=outbreak_seed,
                )

                rows.append({
                    "policy": policy,
                    "run": "" if run is None else run,
                    "graph_seed": graph_seed,
                    "outbreak_seed": outbreak_seed,
                    "patient_zero": result["initial_infected"][0],
                    "infected": result["total_infected"],
                    "attack_fraction": result["total_infected"] / N,
                    "peak_infected": result["peak_infected"],
                    "duration": result["duration"],
                    "ended": result["epidemic_ended"],
                })

    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    """Mean infected per policy on the test scenarios."""

    print(f"\n{'policy':>14} {'mean':>7} {'median':>7} "
          f"{'large%':>7} {'truncated':>9}")

    for policy in ("none", "random", "degree", "evolved", "random_search"):
        infected = np.array(
            [r["infected"] for r in rows if r["policy"] == policy]
        )
        truncated = sum(
            not r["ended"] for r in rows if r["policy"] == policy
        )
        print(f"{policy:>14} {infected.mean():>7.1f} "
              f"{np.median(infected):>7.1f} "
              f"{100 * np.mean(infected > 0.1 * N):>7.0f} {truncated:>9}")

    # Spread between independent runs
    for method in ("evolved", "random_search"):
        runs = sorted({r["run"] for r in rows if r["policy"] == method})
        means = [
            np.mean([r["infected"] for r in rows
                     if r["policy"] == method and r["run"] == run])
            for run in runs
        ]
        print(f"{method}: mean per run from {min(means):.1f} "
              f"to {max(means):.1f} (sd {np.std(means, ddof=1):.1f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="small run to check that everything works")
    args = parser.parse_args()

    config = QUICK if args.quick else FULL
    prefix = "quick_" if args.quick else ""

    os.makedirs("results", exist_ok=True)
    start = time.perf_counter()

    train_graphs = create_networks(config["train_graph_seeds"], n=N, m=M)
    rules, history = run_search(config, train_graphs)
    rows = test_policies(config, rules)

    write_csv(f"results/{prefix}e1_test.csv", rows)
    write_csv(f"results/{prefix}e1_history.csv", history)
    write_csv(f"results/{prefix}e1_rules.csv", [
        {
            "method": r["method"],
            "run": r["run"],
            "w_degree": r["weights"][0],
            "w_neighbour_degree": r["weights"][1],
            "w_clustering": r["weights"][2],
            "train_fitness": r["train_fitness"],
            "evaluations": r["evaluations"],
        }
        for r in rules
    ])

    print_summary(rows)
    print(f"\nRuntime: {time.perf_counter() - start:.0f} s")
    print(f"Saved results/{prefix}e1_test.csv, "
          f"{prefix}e1_rules.csv, {prefix}e1_history.csv")


if __name__ == "__main__":
    main()