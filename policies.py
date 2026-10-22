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

def weighted_policy(G, budget, weights):
    """Select people for vaccination based on weighted network features.
       The evolutionary algorithm will optimize the weights."""
    validate_budget(G, budget)

    # Three weights for each feature
    if len(weights) != 3:
        raise ValueError("Weights must be a list of three values.")

    # Get network features for each person
    features = calculate_features(G)
    # Calculate a weighted score for each person
    scores = {}
    for node, values in features.items():
        # Calculate the weighted score for each person
        score = (
            weights[0] * values[0] + weights[1] * values[1] + weights[2] * values[2]
        )

        scores[node] = score

    # Sort people by their weighted score in descending order
    ranked = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    # Vaccinates the highest scoring individuals based on the budget
    return set(ranked[:budget])