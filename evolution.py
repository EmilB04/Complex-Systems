# Emil and Andreas
"""
Evolutionary optimization of vaccination strategies.

Each candidate contains three weights:
1. Degree
2. Average neighbor degree
3. Clustering coefficient

The weights are used by weighted_policy() to select
individuals for vaccination.

Lower fitness means fewer infections.
"""

import numpy as np

from epidemic import simulate_epidemic
from policies import validate_budget, weighted_policy


def check_weights(weights):
    """Ensure a candidate contains three valid weights."""

    values = np.asarray(weights, dtype=float)

    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(
            "Candidate must have three finite weights."
        )

    if np.any(values < 0) or np.any(values > 1):
        raise ValueError(
            "Candidate weights must be between 0 and 1."
        )

    return values


def create_candidate(rng):
    """
    Create a candidate with three random weights.

    Each weight represents the importance of
    a network feature in the vaccination policy.
    """

    # Generate three weights between 0 and 1
    return rng.uniform(0, 1, size=3)


def create_population(population_size=20, seed=42):
    """
    Create the initial population of weight candidates.

    Each candidate contains exactly three weights.
    """

    if (
        not isinstance(population_size, int)
        or isinstance(population_size, bool)
        or population_size < 1
    ):
        raise ValueError(
            "Population size must be a positive integer."
        )

    rng = np.random.default_rng(seed)

    # Create multiple candidates with different weights
    population = [
        create_candidate(rng)
        for _ in range(population_size)
    ]

    return population


def mutate(candidate, rng, sigma=0.1):
    """
    Mutate a candidate by adding Gaussian noise.

    The original candidate is not modified.
    Weights remain between 0 and 1.
    """

    weights = check_weights(candidate)

    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError(
            "Sigma must be non-negative and finite."
        )

    # Add random noise to each weight
    noise = rng.normal(0, sigma, size=3)

    mutated = weights + noise

    # Keep weights between 0 and 1
    return np.clip(mutated, 0, 1)


def crossover(parent_a, parent_b, rng):
    """
    Combine weights from two parent candidates.

    Each weight is inherited from either parent.
    """

    a = check_weights(parent_a)
    b = check_weights(parent_b)

    # Randomly decide which parent supplies each weight
    mask = rng.random(3) < 0.5

    offspring = np.where(mask, a, b)

    return offspring


def evaluate_fitness(
    G,
    candidate,
    seeds,
    budget,
    beta=0.05,
    gamma=0.10
):
    """
    Evaluate a candidate across several epidemic simulations.

    The candidate contains three weights.
    weighted_policy() selects the vaccinated individuals.

    Lower fitness means fewer infections.
    """

    validate_budget(G, budget)
    weights = check_weights(candidate)

    seeds = list(seeds)
    nodes = sorted(G.nodes())

    if not nodes or not seeds:
        raise ValueError(
            "A nonempty network and at least one seed are required."
        )

    # Use Andreas' weighted policy to select people
    vaccinated = weighted_policy(G, budget, weights)

    infection_counts = []

    for seed in seeds:

        # Choose initial exposure independently of vaccination
        rng = np.random.default_rng(seed)
        patient_zero = int(rng.choice(nodes))

        # Simulate the epidemic using selected individuals
        result = simulate_epidemic(
            G=G,
            vaccinated=vaccinated,
            beta=beta,
            gamma=gamma,
            initial_infected={patient_zero},
            seed=seed
        )

        # Save the number of infected individuals
        infection_counts.append(result["total_infected"])

    # Return average infections across all simulations
    return float(np.mean(infection_counts))


def tournament_selection(
    population,
    fitness_scores,
    rng,
    tournament_size=3
):
    """
    Select a candidate using tournament selection.

    The candidate with the lowest fitness wins.
    """

    if (
        len(population) != len(fitness_scores)
        or not population
    ):
        raise ValueError(
            "Population and scores must have equal nonzero length."
        )

    if (
        not isinstance(tournament_size, int)
        or not 1 <= tournament_size <= len(population)
    ):
        raise ValueError("Invalid tournament size.")

    # Randomly choose candidates for the tournament
    selected_indices = rng.choice(
        len(population),
        size=tournament_size,
        replace=False
    )

    # Find the candidate with the lowest fitness
    winner_index = min(
        selected_indices,
        key=lambda index: fitness_scores[int(index)]
    )

    # Return a copy so the original is not modified
    return population[int(winner_index)].copy()


def run_evolution(
    G,
    budget,
    population_size=20,
    generations=10,
    mutation_rate=0.2,
    tournament_size=3,
    fitness_seeds=None,
    seed=42
):
    """
    Optimize vaccination weights using a genetic algorithm.

    Returns:
        Best weights, vaccinated people,
        best fitness and fitness history.
    """

    # 1. Validate parameters
    validate_budget(G, budget)

    if G.number_of_nodes() == 0:
        raise ValueError("Network cannot be empty.")

    if not isinstance(population_size, int) or population_size < 2:
        raise ValueError(
            "Population size must be at least 2."
        )

    if not isinstance(generations, int) or generations < 1:
        raise ValueError(
            "Generations must be at least 1."
        )

    if not np.isfinite(mutation_rate) or not 0 <= mutation_rate <= 1:
        raise ValueError(
            "Mutation rate must be between 0 and 1."
        )

    if (
        not isinstance(tournament_size, int)
        or not 1 <= tournament_size <= population_size
    ):
        raise ValueError("Invalid tournament size.")

    if fitness_seeds is None:
        fitness_seeds = [10, 20, 30, 40, 50]

    fitness_seeds = list(fitness_seeds)

    if not fitness_seeds:
        raise ValueError(
            "At least one fitness seed is required."
        )

    # 2. Initialize random generator and population
    rng = np.random.default_rng(seed)

    population = create_population(
        population_size=population_size,
        seed=seed
    )

    best_weights = None
    best_fitness = float("inf")
    fitness_history = []

    # Store evaluated candidates to avoid repeated simulations
    fitness_cache = {}

    def get_fitness(weights):
        # Convert the weights into a hashable tuple
        key = tuple(weights.tolist())

        if key not in fitness_cache:
            fitness_cache[key] = evaluate_fitness(
                G=G,
                candidate=weights,
                seeds=fitness_seeds,
                budget=budget
            )

        return fitness_cache[key]

    # 3. Run the genetic algorithm
    for generation in range(generations):

        # Evaluate each candidate in the population
        fitness_scores = [
            get_fitness(weights)
            for weights in population
        ]

        # Find the best candidate in this generation
        best_index = int(np.argmin(fitness_scores))

        generation_best = population[best_index].copy()
        generation_score = fitness_scores[best_index]

        # Update the best solution found so far
        if generation_score < best_fitness:
            best_fitness = generation_score
            best_weights = generation_best.copy()

        # Store statistics for the report
        fitness_history.append({
            "generation": generation + 1,
            "best_fitness": best_fitness,
            "average_fitness": float(np.mean(fitness_scores))
        })

        print(
            f"Generation {generation + 1}/{generations} | "
            f"Best fitness: {best_fitness:.2f} | "
            f"Average fitness: {np.mean(fitness_scores):.2f}"
        )

        # No offspring needed after the final generation
        if generation == generations - 1:
            break

        # 4. Elitism: keep the best candidate found so far
        new_population = [best_weights.copy()]

        # 5. Generate offspring until the population is full
        while len(new_population) < population_size:

            # Select two parents using tournament selection
            parent_a = tournament_selection(
                population,
                fitness_scores,
                rng,
                tournament_size
            )

            parent_b = tournament_selection(
                population,
                fitness_scores,
                rng,
                tournament_size
            )

            # Combine weights from the parents
            offspring = crossover(
                parent_a,
                parent_b,
                rng
            )

            # Mutate the offspring with a given probability
            if rng.random() < mutation_rate:
                offspring = mutate(
                    offspring,
                    rng,
                    sigma=0.1
                )

            new_population.append(offspring)

        # Replace the old population
        population = new_population

    # 6. Convert the best weights into vaccinated individuals
    best_vaccinated = weighted_policy(
        G,
        budget,
        best_weights
    )

    # 7. Return the optimization results
    return {
        "best_candidate": best_weights.copy(),
        "best_weights": best_weights.copy(),
        "best_vaccinated": best_vaccinated,
        "best_fitness": best_fitness,
        "fitness_history": fitness_history,
        "generations": generations,
        "population_size": population_size,
        "evaluations": len(fitness_cache)
    }
