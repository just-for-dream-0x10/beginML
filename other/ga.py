import numpy as np
import matplotlib.pyplot as plt

# Problem Definition
X_BOUND = [-1, 2]  # Search range
DNA_SIZE = 17  # Binary encoding length
POP_SIZE = 100  # Population size
CROSS_RATE = 0.8  # Crossover probability
MUTATION_RATE = 0.01  # Mutation probability
N_GENERATIONS = 200  # Number of generations


def target_function(x):
    """Target function to maximize"""
    return x * np.sin(10 * np.pi * x) + 2.0


def get_fitness(pop, encoding="binary"):
    """Calculate fitness"""
    x = decode_dna(pop, encoding)
    return target_function(x)


def decode_dna(pop, encoding):
    """Decode population to x values"""
    if encoding == "binary":
        # Binary to decimal, then map to [X_BOUND[0], X_BOUND[1]]
        decimal_val = pop.dot(2 ** np.arange(DNA_SIZE)[::-1])
        return X_BOUND[0] + decimal_val * (X_BOUND[1] - X_BOUND[0]) / (2**DNA_SIZE - 1)
    else:  # Real-valued
        return pop


def initialize_population(encoding="binary"):
    """Initialize population"""
    if encoding == "binary":
        return np.random.randint(2, size=(POP_SIZE, DNA_SIZE))
    else:  # Real-valued
        return np.random.uniform(X_BOUND[0], X_BOUND[1], size=(POP_SIZE,))


def select(pop, fitness):
    """Roulette wheel selection"""
    fitness = np.clip(
        fitness - np.min(fitness) + 1e-4, 1e-4, None
    )  # Ensure positive fitness
    idx = np.random.choice(
        np.arange(POP_SIZE), size=POP_SIZE, replace=True, p=fitness / np.sum(fitness)
    )
    return pop[idx]


def crossover(
    parent_chromosome, population_pool, encoding="binary"
):  # Added encoding parameter
    """
    Performs crossover between parent_chromosome and another chromosome from population_pool.
    Returns a new child chromosome.
    """
    child_chromosome = np.copy(
        parent_chromosome
    )  # Start with a copy, especially important for mutable types like numpy arrays

    if np.random.rand() < CROSS_RATE:
        # Select another parent from the population_pool
        idx_other = np.random.randint(0, len(population_pool))
        other_parent_chromosome = population_pool[idx_other]

        if encoding == "binary":
            # parent_chromosome and other_parent_chromosome are binary arrays
            if len(parent_chromosome) > 1:  # Ensure there's a point to cross
                cross_point = np.random.randint(1, len(parent_chromosome))
                child_chromosome[cross_point:] = other_parent_chromosome[cross_point:]
            # If len is 1, child_chromosome remains a copy of parent_chromosome (no crossover possible)
        else:  # Real-valued encoding
            # parent_chromosome and other_parent_chromosome are scalar floats
            alpha = (
                np.random.rand()
            )  # Mixing ratio, could also be a fixed value or from a distribution
            child_chromosome = (
                alpha * parent_chromosome + (1 - alpha) * other_parent_chromosome
            )
            # Note: child_chromosome (which was a copy of a scalar) is reassigned to the new scalar value.

    return child_chromosome  # Return the (potentially modified) child


def mutate(child, encoding="binary"):
    """Mutation"""
    if encoding == "binary":
        for i in range(len(child)):
            if np.random.rand() < MUTATION_RATE:
                child[i] = 1 - child[i]
    else:  # Gaussian mutation for real-valued
        if np.random.rand() < MUTATION_RATE:
            child += np.random.normal(0, 0.1 * (X_BOUND[1] - X_BOUND[0]))
            child = np.clip(child, X_BOUND[0], X_BOUND[1])
    return child


# Main GA Loop
def genetic_algorithm(encoding="binary"):
    pop = initialize_population(encoding)
    best_fitness_history = []

    for generation in range(N_GENERATIONS):
        fitness = get_fitness(pop, encoding)
        best_idx = np.argmax(fitness)
        best_x = decode_dna(pop[best_idx : best_idx + 1], encoding)[0]
        best_fitness = fitness[best_idx]
        best_fitness_history.append(best_fitness)
        print(
            f"Gen {generation:03d} | Best Fitness: {best_fitness:.4f} | Best x: {best_x:.4f}"
        )

        # Evolve population
        new_pop_list = []
        # Elitism: Directly copy the best individual(s)
        num_elite = 1  # Can adjust, e.g., int(0.05 * POP_SIZE)
        if num_elite > 0:
            elite_indices = np.argsort(fitness)[
                -num_elite:
            ]  # Get indices of top individuals
            for elite_idx in elite_indices:
                new_pop_list.append(np.copy(pop[elite_idx]))

        # Fill the rest with offspring
        selected_pop = select(pop, fitness)
        num_offspring = POP_SIZE - num_elite
        for _ in range(num_offspring):
            parent_idx = np.random.randint(
                0, POP_SIZE
            )  # Index for the first parent from selected_pop
            parent_chromosome_for_child = selected_pop[parent_idx]
            child = crossover(
                parent_chromosome_for_child, selected_pop, encoding
            )  # Pass encoding
            child = mutate(child, encoding)
            new_pop_list.append(child)
        pop = np.array(new_pop_list)

    # Final result
    final_fitness = get_fitness(pop, encoding)
    final_best_idx = np.argmax(final_fitness)
    final_best_x = decode_dna(pop[final_best_idx : final_best_idx + 1], encoding)[0]
    print("\n--- Optimization Finished ---")
    print(f"Final Best Fitness: {final_fitness[final_best_idx]:.4f}")
    print(f"Final Best x: {final_best_x:.4f}")
    print("True Global Maximum (approx): f(1.85) ≈ 3.85")

    # Visualization
    x_values = np.linspace(X_BOUND[0], X_BOUND[1], 200)
    y_values = target_function(x_values)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x_values, y_values, label="Target Function")
    plt.scatter(
        final_best_x,
        final_fitness[final_best_idx],
        c="red",
        marker="*",
        s=100,
        label="GA Best Solution",
    )
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.title("Function Optimization")

    plt.subplot(1, 2, 2)
    plt.plot(best_fitness_history, label="Best Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.title("Convergence Over Generations")
    plt.tight_layout()
    plt.savefig("ga_result.png")
    plt.show()


if __name__ == "__main__":
    genetic_algorithm(encoding="binary")  # Try 'real' for real-valued encoding
