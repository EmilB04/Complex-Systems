# Emil
"""
Evolutionary optimization of vaccination strategies.

A candidate is a set of node IDs representing vaccinated individuals.
Every candidate must respect the vaccination budget.

Lower fitness means fewer infections.
"""

import numpy as np

from epidemic import simulate_epidemic


def create_candidate(G, budget, rng):
    """
    Create a random vaccination candidate.

    Args:
        G: NetworkX graph representing the population.
        budget: Number of available vaccines.
        rng: NumPy random number generator.

    Returns:
        set: Node IDs selected for vaccination.
    """
    if not isinstance(budget, int) or not 0 <= budget <= G.number_of_nodes():
        raise ValueError("Invalid vaccination budget.")

    nodes = list(G.nodes())

    selected = rng.choice(
        nodes,
        size=budget,
        replace=False
    )

    return set(selected.tolist())


def create_population(G, budget, population_size=20, seed=42):
    """
    Create the initial population of vaccination candidates.

    Each candidate contains exactly 'budget' unique nodes.
    """
    if not isinstance(population_size, int) or population_size < 1:
        raise ValueError("Population size must be a positive integer.")

    rng = np.random.default_rng(seed)

    population = [
        create_candidate(G, budget, rng)
        for _ in range(population_size)
    ]

    return population


def mutate(candidate, G, rng):
    """
    Replace one vaccinated individual with an unvaccinated one.

    The vaccination budget remains unchanged.
    """
    mutated = candidate.copy()
    available = list(set(G.nodes()) - mutated)

    if not mutated or not available:
        return mutated

    removed = rng.choice(sorted(mutated))
    added = rng.choice(sorted(available))

    mutated.remove(int(removed))
    mutated.add(int(added))

    return mutated


def evaluate_fitness(G, candidate, seeds, beta=0.05, gamma=0.10):
    """
    Evaluate a vaccination strategy across several epidemic runs.

    All candidates use the same initial infection exposure
    for each seed, independently of vaccination status.

    Lower fitness means fewer infections.

    Args:
        G: NetworkX graph representing the population.
        candidate: Set of vaccinated node IDs.
        seeds: Random seeds used for epidemic simulations.
        beta: Infection probability per contact.
        gamma: Recovery probability per time step.

    Returns:
        float: Average number of infected individuals.
    """
    if len(seeds) == 0:
        raise ValueError("At least one seed is required.")

    nodes = sorted(G.nodes())

    if not nodes:
        raise ValueError("The network cannot be empty.")

    infection_counts = []

    for seed in seeds:
        # Choose the initial exposure independently of vaccination.
        rng = np.random.default_rng(seed)
        patient_zero = int(rng.choice(nodes))

        result = simulate_epidemic(
            G=G,
            vaccinated=candidate,
            beta=beta,
            gamma=gamma,
            initial_infected={patient_zero},
            seed=seed
        )

        infection_counts.append(result["total_infected"])

    return float(np.mean(infection_counts))


def tournament_selection(
    population,
    fitness_scores,
    rng,
    tournament_size=3
):
    """
    Select a candidate using tournament selection.

    Lower fitness scores are considered better.
    """
    if len(population) != len(fitness_scores):
        raise ValueError(
            "Population and fitness scores must have equal length."
        )

    if not 1 <= tournament_size <= len(population):
        raise ValueError("Invalid tournament size.")

    # Randomly select candidates for the tournament
    selected_indices = rng.choice(
        len(population),
        size=tournament_size,
        replace=False
    )

    # Select the candidate with the lowest fitness score
    winner_index = min(
        selected_indices,
        key=lambda i: fitness_scores[i]
    )

    return population[int(winner_index)].copy()


def crossover(parent_a, parent_b, G, rng):
    """
    Combine two vaccination candidates into one offspring.

    Both parents must have the same vaccination budget.
    The offspring contains exactly the same number of nodes.
    """
    if len(parent_a) != len(parent_b):
        raise ValueError(
            "Parents must have the same vaccination budget."
        )

    nodes = set(G.nodes())

    if not parent_a.issubset(nodes) or not parent_b.issubset(nodes):
        raise ValueError("Parents contain invalid node IDs.")

    budget = len(parent_a)

    # Preserve nodes shared by both parents.
    offspring = parent_a & parent_b

    # Randomly inherit remaining nodes from either parent.
    remaining = list((parent_a | parent_b) - offspring)
    rng.shuffle(remaining)

    needed = budget - len(offspring)
    offspring.update(remaining[:needed])

    return offspring


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
    Optimize a vaccination strategy using a genetic algorithm.

    Lower fitness means fewer infections.

    Returns:
        dict: Best candidate, best fitness and fitness history.
    """
    # 1. Validate parameters
    if not isinstance(population_size, int) or population_size < 2:
        raise ValueError("Population size must be at least 2.")

    if not isinstance(generations, int) or generations < 1:
        raise ValueError("Generations must be at least 1.")

    if not 0 <= mutation_rate <= 1:
        raise ValueError("Mutation rate must be between 0 and 1.")

    if not 1 <= tournament_size <= population_size:
        raise ValueError("Invalid tournament size.")

    if fitness_seeds is None:
        fitness_seeds = [10, 20, 30, 40, 50]

    fitness_seeds = list(fitness_seeds)

    if not fitness_seeds:
        raise ValueError("At least one fitness seed is required.")

    # 2. Initialize random generator and population
    rng = np.random.default_rng(seed)

    population = create_population(
        G,
        budget=budget,
        population_size=population_size,
        seed=seed
    )

    best_candidate = None
    best_fitness = float("inf")
    fitness_history = []

    # Cache fitness values to avoid repeated simulations
    fitness_cache = {}

    def get_fitness(candidate):
        key = frozenset(candidate)

        if key not in fitness_cache:
            fitness_cache[key] = evaluate_fitness(
                G,
                candidate,
                seeds=fitness_seeds
            )

        return fitness_cache[key]

    # 3. Run the evolution
    for generation in range(generations):

        # Evaluate every candidate
        fitness_scores = [
            get_fitness(candidate)
            for candidate in population
        ]

        # Find the best candidate in this generation
        generation_best_index = int(np.argmin(fitness_scores))
        generation_best_fitness = fitness_scores[generation_best_index]
        generation_best_candidate = population[generation_best_index].copy()

        # Update the best solution found so far
        if generation_best_fitness < best_fitness:
            best_fitness = generation_best_fitness
            best_candidate = generation_best_candidate.copy()

        # Store fitness statistics
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

        # No need to create offspring after the final generation
        if generation == generations - 1:
            break

        # 4. Elitism: preserve the best candidate
        new_population = [generation_best_candidate.copy()]

        # 5. Generate offspring until the population is full
        while len(new_population) < population_size:

            # Select parents
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

            # Combine the parents
            offspring = crossover(
                parent_a,
                parent_b,
                G,
                rng
            )

            # Apply mutation with the specified probability
            if rng.random() < mutation_rate:
                offspring = mutate(offspring, G, rng)

            new_population.append(offspring)

        # Replace the old population
        population = new_population

    # 6. Return optimization results
    return {
        "best_candidate": best_candidate,
        "best_fitness": best_fitness,
        "fitness_history": fitness_history,
        "generations": generations,
        "population_size": population_size,
        "evaluations": len(fitness_cache)
    }



if __name__ == "__main__":
    from network import create_network

    # Create the contact network
    G = create_network(n=200, m=3, seed=42)

    # Number of available vaccines
    budget = 20

    # Generate initial population
    population = create_population(
        G,
        budget=budget,
        population_size=10,
        seed=42
    )

    print("=== Population ===")
    print("Population size:", len(population))
    print("First candidate:", sorted(population[0]))
    print("Vaccinated individuals:", len(population[0]))

    # Test mutation
    rng = np.random.default_rng(123)
    mutated = mutate(population[0], G, rng)

    print("\n=== Mutation ===")
    print("Mutated candidate:", sorted(mutated))
    print("Budget preserved:", len(mutated) == budget)

    # Evaluate the first candidate with fixed exposure scenarios
    fitness = evaluate_fitness(
        G=G,
        candidate=population[0],
        seeds=[10, 20, 30, 40, 50]
    )

    print("\n=== Fitness Evaluation ===")
    print("Candidate fitness:", fitness)

    # Temporary fitness scores for testing tournament selection
    test_fitness = [85, 42, 91, 65, 38, 73, 55, 99, 61, 47]

    selected = tournament_selection(
        population,
        test_fitness,
        rng,
        tournament_size=3
    )

    selected_index = population.index(selected)

    print("\n=== Tournament Selection ===")
    print("Selected candidate:", selected_index)
    print("Selected fitness:", test_fitness[selected_index])
    print("Vaccinated individuals:", len(selected))

    # Test crossover
    parent_a = population[0]
    parent_b = population[1]

    offspring = crossover(
        parent_a,
        parent_b,
        G,
        rng
    )

    print("\n=== Crossover ===")
    print("Parent A:", sorted(parent_a))
    print("Parent B:", sorted(parent_b))
    print("Offspring:", sorted(offspring))
    print("Budget preserved:", len(offspring) == budget)

    # Run the complete genetic algorithm
    print("\n=== Genetic Algorithm ===")

    results = run_evolution(
        G=G,
        budget=20,
        population_size=20,
        generations=15,
        mutation_rate=0.2,
        tournament_size=3,
        fitness_seeds=list(range(100, 130)),
        seed=42
    )

    print("\n=== Evolution Results ===")
    print("Best candidate:", sorted(results["best_candidate"]))
    print("Best fitness:", results["best_fitness"])
    print("Vaccinated individuals:", len(results["best_candidate"]))
    print("Unique candidates evaluated:", results["evaluations"])

    # Evaluate the evolved strategy on unseen epidemic scenarios
    test_fitness = evaluate_fitness(
    G=G,
    candidate=results["best_candidate"],
    seeds=list(range(1000, 1100))
    )

    print("\n=== Generalization Test ===")
    print("Training fitness:", results["best_fitness"])
    print("Test fitness:", test_fitness)