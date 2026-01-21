import random
def generate_mutated_sequence(protein_sequence):
    # Convert the protein sequence to a list for easier manipulation
    mutated_sequence = list(protein_sequence)

    # Randomly decide whether to mutate, insert, or delete an amino acid
    action = random.choice(['mutate', 'insert', 'delete'])

    if action == 'mutate':
        # Randomly mutate one amino acid
        index = random.randint(0, len(mutated_sequence) - 1)
        amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y',
                       'V']
        new_amino_acid = random.choice(amino_acids)
        mutated_sequence[index] = new_amino_acid
    elif action == 'insert':
        # Randomly insert one amino acid at a random position
        index = random.randint(0, len(mutated_sequence))
        amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y',
                       'V']
        new_amino_acid = random.choice(amino_acids)
        mutated_sequence.insert(index, new_amino_acid)
    elif len(mutated_sequence)>=10:
        # Randomly delete one amino acid
        index = random.randint(0, len(mutated_sequence) - 1)
        mutated_sequence.pop(index)

    # Convert the mutated sequence back to a string
    return ''.join(mutated_sequence)


def mutate_crossover(parent1, parent2):
    # Define the mutation rate
    mutation_rate = 0.02

    # Determine the lengths of the parents
    len1, len2 = len(parent1), len(parent2)

    # Create a child sequence that is a combination of both parents
    if random.random() < 0.5:  # Randomly choose which parent to start with
        child = list(parent1)
    else:
        child = list(parent2)

    # Select a crossover point
    crossover_point = random.randint(1, min(len1, len2) - 1)  # Ensure valid crossover point

    # Perform crossover
    child[crossover_point:] = list(parent2)[crossover_point:] if child is parent1 else list(parent1)[crossover_point:]

    # Apply mutation
    for i in range(len(child)):
        if random.random() < mutation_rate:
            amino_acid = random.choice('ACDEFGHIKLMNPQRSTVWY')  # Randomly choose a new amino acid
            child[i] = amino_acid

    return ''.join(child)
