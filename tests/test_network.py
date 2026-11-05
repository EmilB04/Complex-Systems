# Khalid - Tests for network generation

from network import create_networks


def test_create_networks():
    graphs = create_networks(seeds=[1, 2, 3], n=50, m=2)

    # One network per seed, all with the requested size
    assert len(graphs) == 3
    assert all(G.number_of_nodes() == 50 for G in graphs)

    # Different seeds give different networks
    edges = [set(G.edges()) for G in graphs]
    assert edges[0] != edges[1]
    assert edges[1] != edges[2]

    # Same seeds give the same networks
    again = create_networks(seeds=[1, 2, 3], n=50, m=2)
    assert [set(G.edges()) for G in again] == edges