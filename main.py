from network import create_network
from policies import random_policy, degree_policy

# Create social network
G = create_network()

# We have 20 vaccines available
budget = 20

# Test strategies
random_vaccinated = random_policy(G, budget)
degree_vaccinated = degree_policy(G, budget)

# Print results
print("Random vaccination:", sorted(random_vaccinated))
print("Degree vaccination:", sorted(degree_vaccinated))

# Check that each strategy selects 20 people
assert len(random_vaccinated) == budget
assert len(degree_vaccinated) == budget
