# Khalid - Demo: train on several networks and test on new ones

import time

from network import create_networks
from evolution import evaluate_fitness, run_evolution

# Separate networks for training and testing
train_graphs = create_networks(seeds=range(1, 6))       # 5 networks
test_graphs = create_networks(seeds=range(101, 121))    # 20 new networks

# 10 outbreaks per training network = 50 training scenarios
train_seeds = list(range(10, 20))
# 10 outbreaks per test network = 200 test scenarios
test_seeds = list(range(1000, 1010))

start = time.perf_counter()

# Run genetic algorithm
result = run_evolution(
    G=train_graphs,
    budget=20,
    population_size=10,
    generations=20,
    fitness_seeds=train_seeds,
    seed=42
)

print("\n=== Results ===")
print("Best weights:", result["best_weights"])
print("Training fitness:", result["best_fitness"])

# Exactly 20 people should be vaccinated on each network
assert all(len(v) == 20 for v in result["best_vaccinated"])

test_fitness = evaluate_fitness(
    G=test_graphs,
    candidate=result["best_weights"],
    seeds=test_seeds,
    budget=20
)

# Highest-degree-first is the same as weights (1, 0, 0)
degree_fitness = evaluate_fitness(
    G=test_graphs,
    candidate=[1.0, 0.0, 0.0],
    seeds=test_seeds,
    budget=20
)

print("Test fitness (evolved):", test_fitness)
print("Test fitness (degree): ", degree_fitness)
print(f"Runtime: {time.perf_counter() - start:.1f} s")