from __future__ import annotations

import random
from typing import Protocol


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


class RandomLike(Protocol):
    def random(self) -> float: ...
    def randint(self, a: int, b: int) -> int: ...
    def choice(self, seq: str): ...


def mutate_crossover(
    parent1: str,
    parent2: str,
    mutation_rate: float = 0.02,
    rng: RandomLike | None = None,
) -> str:
    """Create one recombinant child and apply independent point mutations.

    The prefix is taken from a randomly chosen base parent and the suffix from
    the other parent.  Passing an explicit ``random.Random`` instance makes the
    operation reproducible without mutating Python's process-wide RNG state.
    """
    if not parent1 or not parent2:
        raise ValueError("Parents must be non-empty sequences")
    if min(len(parent1), len(parent2)) < 2:
        raise ValueError("Parents must contain at least two residues")
    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError("mutation_rate must be between 0 and 1")

    rng = rng or random
    base_is_parent1 = rng.random() < 0.5
    base = parent1 if base_is_parent1 else parent2
    donor = parent2 if base_is_parent1 else parent1
    crossover_point = rng.randint(1, min(len(parent1), len(parent2)) - 1)
    child = list(base[:crossover_point] + donor[crossover_point:])

    for index in range(len(child)):
        if rng.random() < mutation_rate:
            child[index] = rng.choice(AMINO_ACIDS)

    return "".join(child)
