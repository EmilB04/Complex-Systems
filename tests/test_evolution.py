
"""
Unit tests for the genetic algorithm in evolution.py.

Tests cover:
- Candidate generation
- Population generation
- Mutation
- Fitness evaluation
- Tournament selection
- Crossover
- Complete evolutionary optimization
"""

import numpy as np
import pytest

from network import create_network
from evolution import (
    create_candidate,
    create_population,
    mutate,
    evaluate_fitness,
    tournament_selection,
    crossover,
    run_evolution,
)


# --------------------------------------------------
# Candidate generation
# --------------------------------------------------

def test_create_candidate_respects_budget():
    G = create_network(n=50, m=2, seed=42)
    rng = np.random.default_rng(42)

    candidate = create_candidate(G, budget=10, rng=rng)

    assert len(candidate) == 10
    assert candidate.issubset(set(G.nodes()))


def test_create_candidate_invalid_budget():
    G = create_network(n=50, m=2, seed=42)
    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        create_candidate(G, budget=-1, rng=rng)

    with pytest.raises(ValueError):
        create_candidate(G, budget=51, rng=rng)


# --------------------------------------------------
# Population generation
# --------------------------------------------------

def test_population_respects_budget():
    G = create_network(n=200, m=3, seed=42)

    population = create_population(
        G,
        budget=20,
        population_size=10,
        seed=42
    )

    assert len(population) == 10

    for candidate in population:
        assert len(candidate) == 20
        assert candidate.issubset(set(G.nodes()))


def test_population_is_reproducible():
    G = create_network(n=50, m=2, seed=42)

    population_a = create_population(
        G, budget=5, population_size=10, seed=42
    )

    population_b = create_population(
        G, budget=5, population_size=10, seed=42
    )

    assert population_a == population_b


# --------------------------------------------------
# Mutation
# --------------------------------------------------

def test_mutation_preserves_budget():
    G = create_network(n=200, m=3, seed=42)
    rng = np.random.default_rng(42)

    original = create_population(
        G, budget=20, population_size=1
    )[0]

    mutated = mutate(original, G, rng)

    assert len(mutated) == 20
    assert len(original) == 20
    assert mutated.issubset(set(G.nodes()))

    # Exactly one individual must be replaced.
    assert len(original ^ mutated) == 2


def test_mutation_does_not_modify_original():
    G = create_network(n=50, m=2, seed=42)
    rng = np.random.default_rng(42)

    original = {0, 1, 2, 3, 4}
    original_copy = original.copy()

    mutate(original, G, rng)

    assert original == original_copy


# --------------------------------------------------
# Fitness evaluation
# --------------------------------------------------

def test_fitness_with_no_transmission():
    G = create_network(n=50, m=2, seed=42)

    # No vaccinated individuals, so patient zero
    # always becomes infected.
    fitness = evaluate_fitness(
        G=G,
        candidate=set(),
        seeds=[10, 20, 30],
        beta=0.0,
        gamma=1.0
    )

    # No secondary infections are possible.
    assert fitness == 1.0


def test_fitness_with_full_vaccination():
    G = create_network(n=20, m=2, seed=42)

    candidate = set(G.nodes())

    fitness = evaluate_fitness(
        G=G,
        candidate=candidate,
        seeds=[10, 20, 30]
    )

    # Every initial exposure is blocked.
    assert fitness == 0.0


def test_fitness_is_reproducible():
    G = create_network(n=50, m=2, seed=42)
    candidate = {0, 1, 2, 3, 4}

    fitness_a = evaluate_fitness(
        G, candidate, seeds=[10, 20, 30]
    )

    fitness_b = evaluate_fitness(
        G, candidate, seeds=[10, 20, 30]
    )

    assert fitness_a == fitness_b


def test_fitness_requires_seeds():
    G = create_network(n=20, m=2, seed=42)

    with pytest.raises(ValueError):
        evaluate_fitness(G, {0, 1}, seeds=[])


# --------------------------------------------------
# Tournament selection
# --------------------------------------------------

def test_tournament_selection():
    G = create_network(n=30, m=2, seed=42)

    population = create_population(
        G, budget=5, population_size=6, seed=42
    )

    fitness_scores = [30, 20, 10, 40, 50, 60]
    rng = np.random.default_rng(42)

    selected = tournament_selection(
        population,
        fitness_scores,
        rng,
        tournament_size=3
    )

    assert selected in population
    assert len(selected) == 5


def test_tournament_selects_best_when_all_compete():
    population = [
        {0, 1},
        {2, 3},
        {4, 5}
    ]

    fitness_scores = [30, 10, 20]
    rng = np.random.default_rng(42)

    selected = tournament_selection(
        population,
        fitness_scores,
        rng,
        tournament_size=3
    )

    assert selected == {2, 3}


def test_tournament_rejects_invalid_size():
    population = [{0}, {1}, {2}]
    fitness_scores = [3, 2, 1]
    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        tournament_selection(
            population,
            fitness_scores,
            rng,
            tournament_size=4
        )


# --------------------------------------------------
# Crossover
# --------------------------------------------------

def test_crossover_preserves_budget():
    G = create_network(n=200, m=3, seed=42)
    rng = np.random.default_rng(42)

    population = create_population(
        G,
        budget=20,
        population_size=2
    )

    parent_a, parent_b = population

    offspring = crossover(parent_a, parent_b, G, rng)

    assert len(offspring) == 20
    assert offspring.issubset(parent_a | parent_b)
    assert offspring.issubset(set(G.nodes()))

    # Parents must not be modified.
    assert len(parent_a) == 20
    assert len(parent_b) == 20


def test_crossover_preserves_shared_nodes():
    G = create_network(n=20, m=2, seed=42)
    rng = np.random.default_rng(42)

    parent_a = {0, 1, 2, 3}
    parent_b = {0, 1, 4, 5}

    offspring = crossover(parent_a, parent_b, G, rng)

    assert len(offspring) == 4
    assert {0, 1}.issubset(offspring)


def test_crossover_rejects_different_budgets():
    G = create_network(n=20, m=2, seed=42)
    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        crossover({0, 1}, {2, 3, 4}, G, rng)


# --------------------------------------------------
# Complete genetic algorithm
# --------------------------------------------------

def test_evolution_preserves_budget():
    G = create_network(n=30, m=2, seed=42)

    result = run_evolution(
        G=G,
        budget=5,
        population_size=6,
        generations=3,
        fitness_seeds=[10, 20],
        seed=42
    )

    assert len(result["best_candidate"]) == 5
    assert result["best_candidate"].issubset(set(G.nodes()))
    assert len(result["fitness_history"]) == 3


def test_best_fitness_never_increases():
    G = create_network(n=30, m=2, seed=42)

    result = run_evolution(
        G=G,
        budget=5,
        population_size=6,
        generations=3,
        fitness_seeds=[10, 20],
        seed=42
    )

    history = [
        row["best_fitness"]
        for row in result["fitness_history"]
    ]

    # Elitism ensures the best-so-far fitness
    # cannot become worse.
    assert all(
        history[i] <= history[i - 1]
        for i in range(1, len(history))
    )


def test_evolution_is_reproducible():
    G = create_network(n=30, m=2, seed=42)

    kwargs = {
        "G": G,
        "budget": 5,
        "population_size": 6,
        "generations": 3,
        "fitness_seeds": [10, 20],
        "seed": 42
    }

    result_a = run_evolution(**kwargs)
    result_b = run_evolution(**kwargs)

    assert result_a["best_candidate"] == result_b["best_candidate"]
    assert result_a["best_fitness"] == result_b["best_fitness"]
    assert result_a["fitness_history"] == result_b["fitness_history"]


def test_evolution_rejects_invalid_parameters():
    G = create_network(n=30, m=2, seed=42)

    with pytest.raises(ValueError):
        run_evolution(G, budget=5, population_size=1)

    with pytest.raises(ValueError):
        run_evolution(G, budget=5, generations=0)

    with pytest.raises(ValueError):
        run_evolution(G, budget=5, mutation_rate=1.5)
