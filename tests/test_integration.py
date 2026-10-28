from network import create_network
from epidemic import simulate_epidemic
from policies import random_policy, degree_policy, weighted_policy
import networkx as nx

def test_weighted_policy_matches_degree():
    G = nx.star_graph(9)

    # Only degree has weight
    weights = [1.0, 0.0, 0.0]

    vaccinated = weighted_policy(G, 1, weights)

    # Node 0 is connected to every other node
    assert vaccinated == {0}