
import networkx as nx
import numpy as np
import pytest

from epidemic import choose_patient_zero, simulate_epidemic


def test_no_transmission():
    G = nx.path_graph(10)

    result = simulate_epidemic(
        G,
        beta=0.0,
        gamma=1.0,
        initial_infected={0},
        seed=42
    )

    # No new infections are possible.
    assert result["total_infected"] == 1
    assert result["epidemic_ended"] is True


def test_all_vaccinated():
    G = nx.path_graph(10)

    result = simulate_epidemic(
        G,
        vaccinated=set(G.nodes()),
        seed=42
    )

    assert result["total_infected"] == 0
    assert result["vaccinated_count"] == 10


def test_population_is_conserved():
    G = nx.path_graph(20)

    result = simulate_epidemic(
        G,
        vaccinated={0, 1, 2},
        initial_infected={3},
        seed=42
    )

    for row in result["history"]:
        total = (
            row["susceptible"]
            + row["infected"]
            + row["recovered"]
        )
        assert total == G.number_of_nodes()


def test_invalid_infection_probability():
    G = nx.path_graph(10)

    with pytest.raises(ValueError):
        simulate_epidemic(G, beta=1.5)



def test_vaccination_blocks_initial_infection():
    G = nx.path_graph(10)

    result = simulate_epidemic(
        G=G,
        vaccinated={0},
        initial_infected={0},
        beta=1.0,
        gamma=0.1,
        seed=42
    )

    assert result["total_infected"] == 0
    assert result["peak_infected"] == 0
    assert result["epidemic_ended"] is True
    assert result["duration"] == 0

def test_patient_zero_is_never_vaccinated():
    G = nx.path_graph(10)
    vaccinated = set(range(9))

    for seed in range(20):
        rng = np.random.default_rng(seed)
        assert choose_patient_zero(G.nodes(), vaccinated, rng) == 9


def test_patient_zero_when_everyone_is_vaccinated():
    G = nx.path_graph(5)
    rng = np.random.default_rng(42)

    assert choose_patient_zero(G.nodes(), set(G.nodes()), rng) is None


def test_patient_zero_is_shared_across_policies():
    G = nx.path_graph(50)
    policy_a = set(range(0, 10))
    policy_b = set(range(40, 50))

    for seed in range(30):
        a = choose_patient_zero(G.nodes(), policy_a, np.random.default_rng(seed))
        b = choose_patient_zero(G.nodes(), policy_b, np.random.default_rng(seed))
        unvaccinated = choose_patient_zero(G.nodes(), set(), np.random.default_rng(seed))

        # If the first person in the order is unvaccinated under both
        # policies, both policies start the outbreak there.
        if unvaccinated not in policy_a | policy_b:
            assert a == b == unvaccinated


def test_default_outbreak_always_starts():
    G = nx.path_graph(10)

    for seed in range(20):
        result = simulate_epidemic(
            G,
            vaccinated={0, 1, 2, 3, 4},
            beta=0.0,
            gamma=1.0,
            seed=seed
        )

        # One unvaccinated person is always infected at the start
        assert result["total_infected"] == 1


def test_long_outbreak_is_not_truncated():
    # A chain where each step infects exactly one new person.
    # The outbreak needs about 300 steps to reach the end.
    G = nx.path_graph(300)

    result = simulate_epidemic(
        G,
        beta=1.0,
        gamma=1.0,
        initial_infected={0},
        seed=42
    )

    assert result["total_infected"] == 300
    assert result["epidemic_ended"] is True