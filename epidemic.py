# Emil
"""
SIR epidemic simulation on a contact network.

S = Susceptible
I = Infected
R = Recovered or vaccinated (immune)

Vaccinated individuals are immune from the start.
All state changes are applied simultaneously after each time step.

The initial infection is drawn among unvaccinated individuals.
"""

import numpy as np


def choose_patient_zero(nodes, vaccinated, rng):
    """
    Choose the first unvaccinated person in a random order of all nodes.

    The order depends only on rng, not on who is vaccinated. Two policies
    that use the same seed therefore get the same patient zero whenever
    that person is unvaccinated under both policies.

    Returns None if everyone is vaccinated.
    """

    order = rng.permutation(sorted(nodes))

    for node in order:
        if int(node) not in vaccinated:
            return int(node)

    return None


def simulate_epidemic(
    G,
    vaccinated=None,
    beta=0.05,
    gamma=0.10,
    initial_infected=None,
    max_steps=1000,
    seed=42
):
    """
    Simulate epidemic spreading using a discrete-time SIR model.

    Args:
        G: NetworkX graph representing contact relationships.
        vaccinated: Set of node IDs vaccinated before the outbreak.
        beta: Infection probability per infectious contact per step.
        gamma: Recovery probability per infected node per step.
        initial_infected: Initial infected nodes. If None, choose one
        random unvaccinated node.
        max_steps: Maximum number of simulation steps.
        seed: Random seed for reproducibility.

    Returns:
        dict: Epidemic results and history.
    """

    # 1. Validate parameters
    if not 0 <= beta <= 1:
        raise ValueError("beta must be between 0 and 1")

    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be between 0 and 1")

    if not isinstance(max_steps, int) or max_steps < 0:
        raise ValueError("max_steps must be a non-negative integer")

    # Two separate random streams that both depend only on seed:
    # one chooses patient zero, the other drives the epidemic.
    exposure_seed, epidemic_seed = np.random.SeedSequence(seed).spawn(2)
    rng = np.random.default_rng(epidemic_seed)

    nodes = set(G.nodes())
    vaccinated = set() if vaccinated is None else set(vaccinated)

    if not vaccinated.issubset(nodes):
        raise ValueError(
            "Vaccinated nodes must exist in the network"
        )

    # 2. Initialize the population states
    susceptible = nodes - vaccinated
    infected = set()
    recovered = vaccinated.copy()

    # 3. Select the initial infection exposure
    if initial_infected is None:
        # Choose among unvaccinated people only.
        exposure_rng = np.random.default_rng(exposure_seed)
        patient_zero = choose_patient_zero(nodes, vaccinated, exposure_rng)
        initial_infected = set() if patient_zero is None else {patient_zero}
    else:
        initial_infected = set(initial_infected)

    # Check that all initial exposure nodes exist.
    if not initial_infected.issubset(nodes):
        raise ValueError(
            "Initial infected nodes must exist in the network"
        )

    # Vaccination prevents infection, including initial exposure.
    infected = initial_infected - vaccinated
    susceptible -= infected

    # 4. Store initial state (time step 0)
    history = [{
        "step": 0,
        "susceptible": len(susceptible),
        "infected": len(infected),
        "recovered": len(recovered),
    }]

    total_infected = len(infected)
    infected_at_start = set(infected)

    # 5. Run the epidemic simulation
    for step in range(1, max_steps + 1):
        # Stop when no infectious individuals remain.
        if not infected:
            break

        new_infected = set()
        new_recovered = set()

        # Evaluate transmission from currently infected nodes.
        for person in sorted(infected):

            for neighbor in sorted(G.neighbors(person)):
                if neighbor in susceptible:
                    if rng.random() < beta:
                        new_infected.add(neighbor)

            # Infected individuals may recover.
            if rng.random() < gamma:
                new_recovered.add(person)

        # 6. Apply state changes simultaneously
        susceptible -= new_infected
        infected = (infected - new_recovered) | new_infected
        recovered |= new_recovered

        # Count newly infected individuals.
        total_infected += len(new_infected)

        # 7. Record the state for this time step
        history.append({
            "step": step,
            "susceptible": len(susceptible),
            "infected": len(infected),
            "recovered": len(recovered),
        })

    # 8. Return simulation results
    return {
        "total_infected": total_infected,
        "peak_infected": max(
            row["infected"] for row in history
        ),
        "duration": history[-1]["step"],
        "vaccinated_count": len(vaccinated),
        "initial_infected": sorted(infected_at_start),
        "epidemic_ended": len(infected) == 0,
        "history": history,
    }
