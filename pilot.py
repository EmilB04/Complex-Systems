# Khalid - Pilot: choose beta and budget before the main experiments
"""
Runs only the fixed policies (no vaccination, random, degree) for a few
values of beta and budget. The goal is to find a setting where random
vaccination still gives large outbreaks, but degree does not already
stop almost everything. That is where an evolved rule can show a
difference.

Uses its own networks and seeds (200-209 and 5000-5019), which are
never reused in the main experiments.
"""

import csv
import os
import time

import numpy as np

from epidemic import simulate_epidemic
from network import create_networks
from policies import degree_policy, random_policy

BETAS = [0.03, 0.05, 0.08]
BUDGETS = [10, 20, 30]
GAMMA = 0.10

PILOT_GRAPHS = create_networks(seeds=range(200, 210))   # 10 networks
PILOT_SEEDS = list(range(5000, 5020))                   # 20 outbreaks each

LARGE_OUTBREAK = 20     # more than 10 % of 200 people


def run_policy(policy, beta, budget):
    """Return one result row per network x outbreak for a policy."""

    rows = []

    for graph_index, G in enumerate(PILOT_GRAPHS):
        for seed in PILOT_SEEDS:

            if policy == "none":
                vaccinated = set()
            elif policy == "degree":
                vaccinated = degree_policy(G, budget)
            else:
                # New random allocation for every scenario
                vaccinated = random_policy(G, budget, seed=seed)

            result = simulate_epidemic(
                G,
                vaccinated=vaccinated,
                beta=beta,
                gamma=GAMMA,
                seed=seed
            )

            rows.append({
                "infected": result["total_infected"],
                "ended": result["epidemic_ended"],
                "duration": result["duration"],
            })

    return rows


def summarise(rows):
    infected = np.array([r["infected"] for r in rows])

    return {
        "mean": infected.mean(),
        "median": float(np.median(infected)),
        "large_share": float(np.mean(infected > LARGE_OUTBREAK)),
        "truncated": sum(not r["ended"] for r in rows),
        "max_duration": max(r["duration"] for r in rows),
    }


def main():
    os.makedirs("results", exist_ok=True)
    table = []
    start = time.perf_counter()
    simulations = 0

    for beta in BETAS:
        # No vaccination does not depend on the budget
        none = summarise(run_policy("none", beta, 0))
        simulations += len(PILOT_GRAPHS) * len(PILOT_SEEDS)

        for budget in BUDGETS:
            for policy in ("random", "degree"):
                stats = summarise(run_policy(policy, beta, budget))
                simulations += len(PILOT_GRAPHS) * len(PILOT_SEEDS)

                table.append({
                    "beta": beta,
                    "budget": budget,
                    "policy": policy,
                    **{k: round(v, 3) for k, v in stats.items()},
                    "none_mean": round(none["mean"], 3),
                })

    elapsed = time.perf_counter() - start

    with open("results/pilot.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=table[0].keys())
        writer.writeheader()
        writer.writerows(table)

    print(f"{'beta':>5} {'B':>3} {'policy':>7} {'mean':>6} {'median':>6} "
          f"{'large%':>6} {'trunc':>5} {'none':>6}")

    for row in table:
        print(f"{row['beta']:>5} {row['budget']:>3} {row['policy']:>7} "
              f"{row['mean']:>6.1f} {row['median']:>6.1f} "
              f"{100 * row['large_share']:>6.0f} {row['truncated']:>5} "
              f"{row['none_mean']:>6.1f}")

    print(f"\n{simulations} simulations in {elapsed:.1f} s "
          f"({1000 * elapsed / simulations:.2f} ms per simulation)")
    print("Saved to results/pilot.csv")


if __name__ == "__main__":
    main()