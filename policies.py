# Andreas
import networkx as nx
import numpy as np

def validate_budget(G, budget):
    """Ensure the budget is valid for the given network."""
    # Budget must be a whole number (integer)
    if not isinstance(budget, int):
        raise ValueError("Budget must be an integer.")
    # Budget cannot be a negative. 
    if budget < 0:
        raise ValueError("Budget cannot be negative.")
    # Cannot vaccinate more people than exist in the network
    if budget > G.number_of_nodes():
        raise ValueError("Budget cannot exceed the number of people in the network.")

def random_policy(G, budget, seed=42):
    """Select random people for vacination.
       Baseline strategy for vaccination.
    """
    validate_budget(G, budget)
    rng = np.random.default_rng(seed)
    
    # Select people randomly without duplication. 
    selected = rng.choice(
        list(G.nodes()),
        size=budget,
        replace=False
    )
    return set(selected.tolist())

def degree_policy(G, budget):
    """Vacinate people with highest degree(most social contacts)"""
    validate_budget(G, budget)
    # Sort people by number of connections
    ranked = sorted(
        G.nodes(),
        key=lambda node: G.degree(node),
        reverse=True
    )

    # Select the top people based on our budget
    return set(ranked[:budget])


def calculate_features(G):
    """Calculate three network features for each person: 
    1. Degree (number of connections)
    2. Average neighbor degree
    3. Clustering coefficient
    """
    # Get number of contacts (degree) for each person
    degree = dict(G.degree())

    # Calculate how connected each person's neighbors are
    clustering = nx.clustering(G)

    # Find the highest degree to normalize values
    max_degree = max(degree.values(), default=0)

    features = {}

    for node in G.nodes():
        neighbors = list(G.neighbors(node))

        # Normalized degree
        if max_degree > 0:
            d = degree[node] / max_degree
        else: 
            d = 0.0

        # Normalize average neighbor degree
        if neighbors and max_degree > 0:
            avg = sum(degree[n] for n in neighbors) / len(neighbors)
            avg = avg / max_degree
        else:
            avg = 0.0

        # Clustering coefficient
        c = clustering[node]

        # Store the features for this person
        features[node] = [d, avg, c]

    return features

