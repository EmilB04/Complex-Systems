# Khalid - Statistics and figures for the report
"""
Reads the CSV files from experiments.py and produces everything the
report needs. It runs no search, so it is fast and can be rerun at any
time.

Statistics
- Experimental unit: a test network (20 of them). Outbreaks on the same
  network are not independent, so confidence intervals resample whole
  networks (cluster bootstrap), not single outbreaks.
- Evolved and random search are first averaged over their 10 runs on
  each network, so each network counts once per policy.
- Paired differences are computed per network, because every policy
  meets the same scenarios on that network.

Output
- results/summary.csv   one row per policy (table 4)
- results/paired.csv    paired differences with 95 % bootstrap CI
- results/table4.tex    LaTeX rows for table 4
- figures/*.pdf, *.png  outcome distribution, fitness history,
                        rule weights, epidemic curves

Run: python analysis.py
"""

import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")    # write files only, no window
import matplotlib.pyplot as plt
import numpy as np

from epidemic import simulate_epidemic
from evolution import evaluate_fitness
from experiments import BETA, BUDGET, FULL, GAMMA, M, N
from network import create_networks
from policies import degree_policy, random_policy, weighted_policy

RESULTS = "results"
FIGURES = "figures"

BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 0
LARGE_OUTBREAK = 0.1 * N     # more than 20 of 200 people

POLICIES = ["none", "random", "degree", "evolved", "random_search"]
LABELS = {
    "none": "No vaccination",
    "random": "Random",
    "degree": "Highest degree",
    "evolved": "Evolved (GA)",
    "random_search": "Random search",
}

# Colours: the three rules being compared get the three validated
# categorical colours; the two references are neutral grey.
COLOURS = {
    "evolved": "#2a78d6",
    "degree": "#eb6834",
    "random_search": "#1baf7a",
    "random": "#8a8985",
    "none": "#b5b4ae",
}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


# ---------- Reading data ----------

def read_csv(name):
    with open(os.path.join(RESULTS, name), newline="") as f:
        return list(csv.DictReader(f))


def per_network_means(rows):
    """
    Mean infected per (policy, network).

    For evolved and random search this is the mean over all 10 runs and
    all outbreaks on that network, so each network counts once.
    """

    values = defaultdict(list)

    for r in rows:
        values[(r["policy"], int(r["graph_seed"]))].append(float(r["infected"]))

    networks = sorted({key[1] for key in values})

    return networks, {
        policy: np.array([np.mean(values[(policy, g)]) for g in networks])
        for policy in POLICIES
    }


def bootstrap_ci(per_network, rng):
    """95 % CI for the mean by resampling networks with replacement."""

    n = len(per_network)
    samples = rng.integers(0, n, size=(BOOTSTRAP_SAMPLES, n))
    means = per_network[samples].mean(axis=1)

    return np.percentile(means, [2.5, 97.5])


# ---------- Tables ----------

def summary_table(rows, net_means):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    table = []

    for policy in POLICIES:
        selected = [r for r in rows if r["policy"] == policy]
        infected = np.array([float(r["infected"]) for r in selected])
        low, high = bootstrap_ci(net_means[policy], rng)

        table.append({
            "policy": policy,
            "scenarios": len(selected),
            "mean_infected": infected.mean(),
            "ci_low": low,
            "ci_high": high,
            "sd_infected": infected.std(ddof=1),
            "median_infected": float(np.median(infected)),
            "large_share": float(np.mean(infected > LARGE_OUTBREAK)),
            "mean_peak": np.mean([float(r["peak_infected"]) for r in selected]),
            "mean_duration": np.mean([float(r["duration"]) for r in selected]),
            "truncated": sum(r["ended"] != "True" for r in selected),
        })

    return table


def paired_table(net_means):
    """Per-network differences; positive = fewer infections for the second."""

    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    comparisons = [
        ("random", "evolved"),
        ("degree", "evolved"),
        ("random_search", "evolved"),
        ("random", "degree"),
    ]
    table = []

    for first, second in comparisons:
        diff = net_means[first] - net_means[second]
        low, high = bootstrap_ci(diff, rng)

        table.append({
            "comparison": f"{first} - {second}",
            "mean_difference": diff.mean(),
            "ci_low": low,
            "ci_high": high,
            "networks_where_second_better": int(np.sum(diff > 0)),
            "networks": len(diff),
        })

    return table


def run_spread(rows):
    """Mean infected per independent run, for evolved and random search."""

    spread = {}

    for method in ("evolved", "random_search"):
        runs = defaultdict(list)

        for r in rows:
            if r["policy"] == method:
                runs[r["run"]].append(float(r["infected"]))

        spread[method] = np.array([np.mean(v) for v in runs.values()])

    return spread


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()

        for row in rows:
            writer.writerow({
                k: round(v, 3) if isinstance(v, float) else v
                for k, v in row.items()
            })


def write_latex(summary, path):
    """Rows for table 4. Mean with 95 % CI, median and large outbreaks."""

    lines = [
        "% Generated by analysis.py - do not edit by hand.",
        "% Policy & Mean infected [95% CI] & Median & Large outbreaks (%) \\\\",
    ]

    for row in summary:
        lines.append(
            f"{LABELS[row['policy']]} & "
            f"{row['mean_infected']:.1f} "
            f"[{row['ci_low']:.1f}, {row['ci_high']:.1f}] & "
            f"{row['median_infected']:.0f} & "
            f"{100 * row['large_share']:.0f} \\\\"
        )

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------- Figures ----------

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGURES, f"{name}.{ext}"))
    plt.close(fig)


def figure_outcomes(rows, summary):
    """Distribution of outbreak sizes per policy (small multiples)."""

    fig, axes = plt.subplots(
        len(POLICIES), 1, figsize=(5.5, 6.0), sharex=True, sharey=True
    )
    bins = np.arange(0, N + 10, 10)
    means = {row["policy"]: row["mean_infected"] for row in summary}

    for ax, policy in zip(axes, POLICIES):
        infected = [float(r["infected"]) for r in rows if r["policy"] == policy]
        # Show shares, so pooled runs (4000 rows) compare with 400 rows
        weights = np.full(len(infected), 100 / len(infected))

        ax.hist(infected, bins=bins, weights=weights,
                color=COLOURS[policy], rwidth=0.85)
        ax.axvline(means[policy], color=INK, linewidth=1)
        ax.text(means[policy] + 3, 70, f"mean {means[policy]:.1f}",
                color=INK, fontsize=8, va="top")
        ax.set_ylabel(LABELS[policy], rotation=0, ha="right",
                      va="center", color=INK)
        ax.set_ylim(0, 80)
        ax.set_yticks([0, 40, 80])
        ax.grid(axis="x", visible=False)

    axes[-1].set_xlabel("People infected (of 200)")
    fig.text(0.98, 0.5, "Share of test scenarios (%)", rotation=270,
             va="center", ha="center", color=INK_SECONDARY)
    fig.suptitle("Outbreak size on 400 held-out scenarios",
                 fontweight="bold", fontsize=10)
    fig.tight_layout(rect=(0, 0, 0.96, 1))
    save(fig, "fig_outcomes")


def figure_fitness(history, degree_train):
    """Best-so-far training fitness per generation for every GA run."""

    runs = defaultdict(list)

    for r in history:
        runs[r["run"]].append(
            (int(r["generation"]), float(r["best_fitness"]))
        )

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    curves = []

    for points in runs.values():
        points.sort()
        generations = [g for g, _ in points]
        best = [b for _, b in points]
        curves.append(best)
        ax.plot(generations, best, color=COLOURS["evolved"],
                alpha=0.25, linewidth=1)

    median = np.median(np.array(curves), axis=0)
    ax.plot(generations, median, color=COLOURS["evolved"],
            linewidth=2, label="GA, median of 10 runs")
    ax.axhline(degree_train, color=COLOURS["degree"], linewidth=2,
               label="Highest degree")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Training fitness\n(mean infected, lower is better)")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_title("GA training fitness (thin lines: individual runs)")
    ax.legend(loc="upper right")
    save(fig, "fig_fitness")


def figure_weights(rules):
    """Weights of the final rule for every run, scaled to max |w| = 1."""

    features = ["w_degree", "w_neighbour_degree", "w_clustering"]
    names = ["Degree", "Mean neighbour\ndegree", "Clustering"]
    offsets = {"evolved": -0.12, "random_search": 0.12}

    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    ax.axvline(0, color=INK_SECONDARY, linewidth=1)

    for method, offset in offsets.items():
        for i, feature in enumerate(features):
            values = []

            for r in rules:
                if r["method"] != method:
                    continue

                w = np.array([float(r[f]) for f in features])
                # Ranking only depends on the direction of w
                values.append(w[i] / np.abs(w).max())

            ax.scatter(values, np.full(len(values), i + offset),
                       s=30, color=COLOURS[method],
                       edgecolor="white", linewidth=1,
                       label=LABELS[method] if i == 0 else None)

    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(names, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(-1.1, 1.1)
    ax.set_xlabel("Weight (scaled so the largest |weight| is 1)")
    ax.set_title("Final rules from 10 runs of each search")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    save(fig, "fig_weights")


def figure_curves(rules, rows):
    """
    Mean number of infectious people over time on the test scenarios.

    Uses the evolved rule whose test mean is the median of the 10 runs,
    so the figure shows a typical run, not the best one.
    """

    spread = defaultdict(list)

    for r in rows:
        if r["policy"] == "evolved":
            spread[r["run"]].append(float(r["infected"]))

    run_means = {run: np.mean(v) for run, v in spread.items()}
    ordered = sorted(run_means, key=run_means.get)
    typical = ordered[len(ordered) // 2]

    weights = next(
        [float(r["w_degree"]), float(r["w_neighbour_degree"]),
         float(r["w_clustering"])]
        for r in rules
        if r["method"] == "evolved" and r["run"] == typical
    )

    length = 150
    curves = {p: [] for p in ("none", "random", "degree", "evolved")}

    for G in create_networks(FULL["test_graph_seeds"], n=N, m=M):
        fixed = {
            "none": set(),
            "degree": degree_policy(G, BUDGET),
            "evolved": weighted_policy(G, BUDGET, weights),
        }

        for seed in FULL["test_outbreak_seeds"]:
            allocations = dict(fixed)
            allocations["random"] = random_policy(G, BUDGET, seed=seed)

            for policy, vaccinated in allocations.items():
                result = simulate_epidemic(
                    G, vaccinated=vaccinated, beta=BETA, gamma=GAMMA,
                    seed=seed
                )
                infected = [h["infected"] for h in result["history"]]
                # After the outbreak ends, nobody is infectious
                infected = (infected + [0] * length)[:length]
                curves[policy].append(infected)

    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    for policy in ("none", "random", "degree", "evolved"):
        mean = np.mean(curves[policy], axis=0)
        # Degree is drawn wider underneath, since the two curves overlap
        width = 4 if policy == "degree" else 2
        ax.plot(range(length), mean, color=COLOURS[policy],
                linewidth=width, label=LABELS[policy])

    ax.annotate("Highest degree and evolved\nalmost coincide",
                xy=(40, 5.5), xytext=(60, 20), color=INK, fontsize=8,
                arrowprops={"arrowstyle": "-", "color": INK_SECONDARY,
                            "linewidth": 1})

    ax.set_xlabel("Time step")
    ax.set_ylabel("Infectious people (mean)")
    ax.set_xlim(0, length - 1)
    ax.set_ylim(bottom=0)
    ax.set_title(f"Epidemic curves on 400 test scenarios "
                 f"(evolved rule: run {typical})")
    ax.legend(loc="upper right")
    save(fig, "fig_curves")

    return typical


# ---------- Main ----------

def main():
    os.makedirs(FIGURES, exist_ok=True)

    rows = read_csv("e1_test.csv")
    rules = read_csv("e1_rules.csv")
    history = read_csv("e1_history.csv")

    networks, net_means = per_network_means(rows)
    summary = summary_table(rows, net_means)
    paired = paired_table(net_means)
    spread = run_spread(rows)

    write_csv(os.path.join(RESULTS, "summary.csv"), summary)
    write_csv(os.path.join(RESULTS, "paired.csv"), paired)
    write_latex(summary, os.path.join(RESULTS, "table4.tex"))

    # Degree on the training scenarios, for the fitness figure
    train_graphs = create_networks(FULL["train_graph_seeds"], n=N, m=M)
    degree_train = evaluate_fitness(
        train_graphs, [1.0, 0.0, 0.0], FULL["train_outbreak_seeds"],
        BUDGET, beta=BETA, gamma=GAMMA
    )

    figure_outcomes(rows, summary)
    figure_fitness(history, degree_train)
    figure_weights(rules)
    typical = figure_curves(rules, rows)

    print(f"Test networks: {len(networks)}   "
          f"bootstrap samples: {BOOTSTRAP_SAMPLES}\n")
    print(f"{'policy':>14} {'mean':>6} {'95% CI':>15} {'median':>6} "
          f"{'large%':>6} {'peak':>6}")

    for r in summary:
        print(f"{r['policy']:>14} {r['mean_infected']:>6.1f} "
              f"[{r['ci_low']:>5.1f}, {r['ci_high']:>5.1f}] "
              f"{r['median_infected']:>6.0f} "
              f"{100 * r['large_share']:>6.0f} {r['mean_peak']:>6.1f}")

    print("\nPaired differences per network (positive = fewer infections "
          "for the second policy)")

    for r in paired:
        print(f"{r['comparison']:>24}: {r['mean_difference']:>6.1f} "
              f"[{r['ci_low']:.1f}, {r['ci_high']:.1f}]  "
              f"second better on {r['networks_where_second_better']}"
              f"/{r['networks']} networks")

    for method, values in spread.items():
        print(f"\n{method}: mean per run {values.min():.1f}-{values.max():.1f}"
              f", sd between runs {values.std(ddof=1):.1f}")

    print(f"\nDegree training fitness: {degree_train:.2f}")
    print(f"Typical evolved run used in fig_curves: {typical}")
    print("\nSaved results/summary.csv, results/paired.csv, "
          "results/table4.tex and figures/")


if __name__ == "__main__":
    main()