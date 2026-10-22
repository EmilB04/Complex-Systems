
import networkx as nx
import pytest

from epidemic import simulate_epidemic


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
