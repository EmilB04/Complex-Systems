"""
Runs one outbreak on one network for no vaccination, random and
highest-degree-first, using the same settings as the main experiments.
This is a sanity check, not a comparison: use experiments.py and
analysis.py for results.
"""

from epidemic import simulate_epidemic
from experiments import BETA, BUDGET, GAMMA, M, N
from network import create_network
from policies import degree_policy, random_policy


def main():
    G = create_network(n=N, m=M, seed=42)

    policies = {
        "No vaccination": set(),
        "Random": random_policy(G, BUDGET, seed=42),
        "Highest degree": degree_policy(G, BUDGET),
    }

    for name, vaccinated in policies.items():
        # Same seed for every policy, so the outbreak starts in the same
        # person whenever that person is unvaccinated
        result = simulate_epidemic(
            G, vaccinated=vaccinated, beta=BETA, gamma=GAMMA, seed=42
        )

        print(f"=== {name} ===")
        print("Patient zero:", result["initial_infected"])
        print("Total infected:", result["total_infected"])
        print("Peak infected:", result["peak_infected"])
        print("Duration:", result["duration"])
        print()


if __name__ == "__main__":
    main()