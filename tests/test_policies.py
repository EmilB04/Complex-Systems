import pytest 
import networkx as nx
from policies import validate_budget, random_policy, degree_policy, calculate_features, weighted_policy
    


# Test valid vaccination budget
def test_valid_budget():
    G = nx.path_graph(10)

    # These budgets should not raise an error
    validate_budget(G, 0)
    validate_budget(G, 5)
    validate_budget(G, 10)

# Test invalid vaccination budget
def test_invalid_budget():
    G = nx.path_graph(10)

    # Negative budget should raise an error
    with pytest.raises(ValueError):
        validate_budget(G, -1)

    # Budget larger than the network size should raise an error
    with pytest.raises(ValueError):
        validate_budget(G, 11)

    # Budget must be an integer
    with pytest.raises(ValueError):
        validate_budget(G, 5.5)

# Test random vaccination
def test_random_policy_vaccination():
    G = nx.path_graph(10)
    vaccinated = random_policy(G, 3, seed=42)

    # Exactly 3 nodes/people should be vaccinated
    assert len(vaccinated) == 3
    # All selected people/nodes should exist in the graph/network
    assert all(node in G.nodes for node in vaccinated)

# Test random repetition with the same seed
def test_random_reproducibility():
    G = nx.path_graph(10)
    vaccinated_first = random_policy(G, 3, seed=42)
    vaccinated_second = random_policy(G, 3, seed=42)

    # The two vaccination results with the same seed should be identical
    assert vaccinated_first == vaccinated_second

# Test degree-based vaccination
def test_degree_policy():
    # Create star network where node 0 is connected to all other nodes
    G = nx.star_graph(9)

    vaccinated = degree_policy(G, 1)

    # The node with the highest degree (node 0) should be vaccinated
    assert vaccinated == {0}

# Test zero vaccination budget
def test_zero_budget():
    G = nx.path_graph(10)
    

    # No people/nodes should be vaccinated
    assert random_policy(G, 0) == set()
    assert degree_policy(G, 0) == set()


def test_calculate_features():
    # Create a small network for testing
    G = nx.path_graph(3)

    features = calculate_features(G)

    # All three persons have features 
    assert len(features) == 3

    # Each person should have exactly three features
    for values in features.values():
        assert len(values) == 3

        # Each value should be between 0 and 1
        assert all(0 <= value <= 1 for value in values)

    # Perosn 1 has the highest degrees (2 contacts)
    assert features[1][0] == 1.0

    # Person 0 has one contact
    assert features[0][0] == 0.5


def test_weighted_policy():
    G = nx.star_graph(10)

    vaccinated = weighted_policy(G, 3, [0.5, 0.3, 0.2])

    assert len(vaccinated) == 3
    assert vaccinated.issubset(set(G.nodes()))

def test_weighted_policy_zero_budget():
    G = nx.path_graph(10)

    assert weighted_policy(G, 0, [1, 0, 0]) == set()

def test_weighted_policy_invalid_weights():
    G = nx.path_graph(10)

    with pytest.raises(ValueError):
        weighted_policy(G, 3, [0.5, 0.5])