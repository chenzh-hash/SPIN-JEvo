import random


def mutate_crossover(parent1, parent2):
    """Single-point crossover followed by point mutations."""
    mutation_rate = 0.02
    len1, len2 = len(parent1), len(parent2)

    if random.random() < 0.5:
        child = list(parent1)
    else:
        child = list(parent2)

    crossover_point = random.randint(1, min(len1, len2) - 1)
    child[crossover_point:] = list(parent2)[crossover_point:] if child is parent1 else list(parent1)[crossover_point:]

    for i in range(len(child)):
        if random.random() < mutation_rate:
            child[i] = random.choice("ACDEFGHIKLMNPQRSTVWY")

    return "".join(child)
