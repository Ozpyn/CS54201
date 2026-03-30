# Write a program for hidden Markov model that calculates most probable path based upon the given observation. The program should take as input a sequence of probable observations. The graph size should be parameterized, and the weights of the graphs should be generated using a random number generator or read from a file. 

import numpy as np
import itertools

EPSILON = 1e-12

def random_stochastic_matrix(num_rows, num_cols):
    matrix = np.random.rand(num_rows, num_cols)
    return matrix / matrix.sum(axis=1, keepdims=True)

def random_initial_probabilities(num_states):
    vector = np.random.rand(num_states)
    return vector / vector.sum()

def is_valid_emission_sequence(observation_sequence, emission_symbols):
    return all(e in emission_symbols for e in observation_sequence)

def index_in_list(item, lst):
    return lst.index(item)

def states_for_emission(observation, emission_symbols, emission_matrix, num_states):
    emission_index = index_in_list(observation, emission_symbols)
    possible_states = []
    for state in range(num_states):
        if emission_matrix[state, emission_index] >= EPSILON:
            possible_states.append(state)
    return possible_states

def is_valid_state_path(state_sequence, transition_matrix, initial_probabilities):
    if initial_probabilities[state_sequence[0]] < EPSILON:
        return False
    for i in range(len(state_sequence) - 1):
        if transition_matrix[state_sequence[i], state_sequence[i+1]] < EPSILON:
            return False
    return True

def path_probability(state_sequence, observation_sequence, initial_probabilities, transition_matrix, emission_matrix, emission_symbols):
    prob = initial_probabilities[state_sequence[0]] * emission_matrix[state_sequence[0], index_in_list(observation_sequence[0], emission_symbols)]
    for i in range(1, len(state_sequence)):
        prob *= transition_matrix[state_sequence[i-1], state_sequence[i]]
        prob *= emission_matrix[state_sequence[i], index_in_list(observation_sequence[i], emission_symbols)]
    return prob

def hmm_path(states, emission_symbols, transition_matrix, emission_matrix, initial_probabilities, observation_sequence):
    if not is_valid_emission_sequence(observation_sequence, emission_symbols):
        print("Invalid emission sequence!")
        return None

    num_states = len(states)
    
    possible_states_per_observation = [
        states_for_emission(obs, emission_symbols, emission_matrix, num_states) 
        for obs in observation_sequence
    ]

    all_possible_paths = list(itertools.product(*possible_states_per_observation))
    
    max_probability = 0
    best_state_sequence = None

    for path in all_possible_paths:
        if is_valid_state_path(path, transition_matrix, initial_probabilities):
            prob = path_probability(path, observation_sequence, initial_probabilities, transition_matrix, emission_matrix, emission_symbols)
            if prob > max_probability:
                max_probability = prob
                best_state_sequence = path
    
    if best_state_sequence is not None:
        best_state_labels = [states[i] for i in best_state_sequence]
        print(f"Most probable state sequence: {best_state_labels} with probability {max_probability}")
        return best_state_labels, max_probability
    else:
        print("No valid state path found.")
        return None

num_states = 3
num_emissions = 2

states = ['A', 'B', 'C']
emission_symbols = ['x', 'y']
    
transition_matrix = random_stochastic_matrix(num_states, num_states)
emission_matrix = random_stochastic_matrix(num_states, num_emissions)
initial_probabilities = random_initial_probabilities(num_states)
    
print("Transition matrix:\n", transition_matrix)
print("Emission matrix:\n", emission_matrix)
print("Initial probabilities:\n", initial_probabilities)
    
observation_sequence = ['x', 'y', 'x']
hmm_path(states, emission_symbols, transition_matrix, emission_matrix, initial_probabilities, observation_sequence)