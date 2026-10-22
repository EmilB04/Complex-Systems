from network import create_network
from policies import random_policy, degree_policy
from epidemic import simulate_epidemic


def main():
    # Create a network of 200 individuals.
    G = create_network(n=200, m=3, seed=42)

    budget = 20

    # Create vaccination strategies.
    random_vaccinated = random_policy(G, budget, seed=42)
    degree_vaccinated = degree_policy(G, budget)

    # Run both strategies under the same epidemic parameters.
    random_result = simulate_epidemic(
        G,
        vaccinated=random_vaccinated,
        beta=0.05,
        gamma=0.10,
        seed=42
    )

    degree_result = simulate_epidemic(
        G,
        vaccinated=degree_vaccinated,
        beta=0.05,
        gamma=0.10,
        seed=42
    )

    print("=== Random Vaccination ===")
    print("Total infected:", random_result["total_infected"])
    print("Peak infected:", random_result["peak_infected"])
    print("Duration:", random_result["duration"])

    print("\n=== Degree Vaccination ===")
    print("Total infected:", degree_result["total_infected"])
    print("Peak infected:", degree_result["peak_infected"])
    print("Duration:", degree_result["duration"])

if __name__ == "__main__":
    main()

