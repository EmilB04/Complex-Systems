
from network import create_network
from evolution import evaluate_fitness, run_evolution

# Create contact network
G = create_network(n=200, m=3, seed=42)

# Run genetic algorithm
result = run_evolution(
    G=G,
    budget=20,
    population_size=10,
    generations=20,
    fitness_seeds=[10, 20, 30],
    seed=42
)

print("\n=== Results ===")

print("Best weights:", result["best_weights"])
print("Best fitness:", result["best_fitness"])
print("Vaccinated people:", sorted(result["best_vaccinated"]))

# Exactly 20 people should be vaccinated
assert len(result["best_vaccinated"]) == 20

test_fitness = evaluate_fitness(
    G=G,
    candidate=result["best_weights"],
    seeds=list(range(1000, 1100)),
    budget=20
)

print("Training fitness:", result["best_fitness"])
print("Test fitness:", test_fitness)
