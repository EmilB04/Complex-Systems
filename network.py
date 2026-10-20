# Andreas - Network generation using Barabási Albert model
import networkx as nx

def create_network(n=200, m=3, seed=42):
    """Creates a Barabási Albert network to represent social contacts.
    Between people in an epidemic simulation
    
    Parameters:
    n (int): Number of nodes (people) in the network.
    m (int): Number of edges to attach from a new node to existing nodes.
    seed (int): Random seed for reproducibility.

    Returns:
    networkx.Graph: A Barabási Albert network representing social contacts.
    """

    # Ensure m is valid before creating the network.
    # Each node must attach to at least one existing node and fewer than n existing nodes.
    # m must be smaller than the total number of nodes (n).
    if not 1 <= m < n:
        raise ValueError("m must be greater than 0 and less than n")

    # Create the Barabási Albert network representing social contacts.
    # Some nodes will have more connections than others
    # which reflects the heterogeneity of social contacts in real life.
    return nx.barabasi_albert_graph(n, m, seed=seed)

# Example usage of the network generation function.
# This allows us to test the network generation function independently.
if __name__ == "__main__":
    # Create the network using the default parameters.
    G = create_network()

    # Print basic information about the generated network.
    print("Number of people: ", G.number_of_nodes())
    print("Number of contacts: ", G.number_of_edges())
    # Degree = number of direct contacts a person has.
    # This will later be used to prioritize vaccination.
    print("Degree of person 0: ", G.degree(0))

