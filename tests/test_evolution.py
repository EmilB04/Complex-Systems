
# Andreas/Emil - Tests for weighted evolution

import numpy as np
import pytest
import networkx as nx

from policies import weighted_policy
from evolution import (
    check_weights,
    create_candidate,
    create_population,
    mutate,
    crossover,
    evaluate_fitness,
    tournament_selection,
    run_evolution
)


def test_create_candidate():
    rng = np.random.default_rng(42)

    candidate = create_candidate(rng)

    # Each candidate must contain exactly three weights
    assert candidate.shape == (3,)

    # All weights must be between -1 and 1
    assert np.all((candidate >= -1) & (candidate <= 1))


def test_create_population():
    first = create_population(5, seed=42)
    second = create_population(5, seed=42)

    # Population contains five candidates
    assert len(first) == 5

    # Same seed must create identical candidates
    for a, b in zip(first, second):
        assert np.array_equal(a, b)


def test_mutation():
    rng = np.random.default_rng(42)

    original = np.array([0.5, 0.3, 0.2])

    mutated = mutate(original, rng)

    # Mutated candidate must contain three weights
    assert mutated.shape == (3,)

    # Weights must remain within range
    assert np.all((mutated >= -1) & (mutated <= 1))

    # Original candidate must not be modified
    assert np.array_equal(original, [0.5, 0.3, 0.2])


def test_crossover():
    rng = np.random.default_rng(42)

    parent_a = np.array([0.1, 0.2, 0.3])
    parent_b = np.array([0.8, 0.9, 0.7])

    offspring = crossover(parent_a, parent_b, rng)

    # Offspring must inherit weights from its parents
    for i in range(3):
        assert offspring[i] in (parent_a[i], parent_b[i])


def test_tournament_selection():
    rng = np.random.default_rng(42)

    population = [
        np.array([0.1, 0.1, 0.1]),
        np.array([0.2, 0.2, 0.2]),
        np.array([0.3, 0.3, 0.3])
    ]

    fitness_scores = [30, 10, 20]

    # All candidates compete, so the best should win
    selected = tournament_selection(
        population,
        fitness_scores,
        rng,
        tournament_size=3
    )

    assert np.array_equal(selected, population[1])


def test_fitness():
    G = nx.path_graph(12)

    weights = [0.5, 0.3, 0.2]

    first = evaluate_fitness(
        G, weights, seeds=[10, 20], budget=3
    )

    second = evaluate_fitness(
        G, weights, seeds=[10, 20], budget=3
    )

    # Same setup should produce the same result
    assert first == second

    # With beta=0, only the initial person can be infected
    no_transmission = evaluate_fitness(
        G, weights, seeds=[10, 20], budget=3, beta=0
    )

    # Patient zero is never vaccinated, so exactly one infection
    assert no_transmission == 1

def test_full_evolution():
    G = nx.barabasi_albert_graph(30, 2, seed=42)

    settings = {
        "G": G,
        "budget": 5,
        "population_size": 6,
        "generations": 3,
        "fitness_seeds": [10, 20],
        "seed": 42
    }

    first = run_evolution(**settings)
    second = run_evolution(**settings)

    # Exactly five people should be vaccinated
    assert len(first["best_vaccinated"]) == 5

    # Each candidate contains three weights
    assert len(first["best_weights"]) == 3

    # Running twice with the same seed is reproducible
    assert np.array_equal(
        first["best_weights"],
        second["best_weights"]
    )

    assert first["best_fitness"] == second["best_fitness"]

    # Best fitness found so far must never increase
    history = [
        row["best_fitness"]
        for row in first["fitness_history"]
    ]

    assert history == sorted(history, reverse=True)


def test_invalid_parameters():
    G = nx.path_graph(10)

    # Cannot vaccinate more people than exist
    with pytest.raises(ValueError):
        run_evolution(G, budget=11)

    # A candidate must contain exactly three weights
    with pytest.raises(ValueError):
        evaluate_fitness(
            G,
            candidate=[0.1, 0.2],
            seeds=[1],
            budget=3
        )

def test_negative_weights_are_allowed():
    # Negative weights are valid and are used as given
    assert np.array_equal(
        check_weights([-1.0, 0.0, 1.0]),
        [-1.0, 0.0, 1.0]
    )

    # Weights outside [-1, 1] are rejected
    with pytest.raises(ValueError):
        check_weights([1.5, 0.0, 0.0])

    with pytest.raises(ValueError):
        check_weights([-1.5, 0.0, 0.0])


def test_negative_weight_prefers_low_feature_values():
    # Two triangles joined by one edge: nodes 2 and 3 have clustering
    # below 1, the others have clustering 1.
    G = nx.Graph([(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (4, 5), (3, 5)])

    # Only clustering has weight, and it is negative
    vaccinated = weighted_policy(G, 2, [0.0, 0.0, -1.0])

    assert vaccinated == {2, 3}