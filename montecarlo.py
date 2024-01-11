import numpy as np
import random
import csv

# Define the environment
n_rows = 3
n_cols = 3
n_states = n_rows * n_cols
n_actions = 4  # Up, Down, Left, Right

# Define rewards
rewards = np.ones(n_states)
rewards[2] = -5
rewards[4] = -5
rewards[8] = 10  # Goal state

# Value function initialization
V = np.zeros(n_states)

# Monte Carlo parameters
gamma = 0.99  # Discount factor
alpha = 0.1  # Learning rate

def calculate_next_state(current_state, action):
    row, col = current_state // n_cols, current_state % n_cols

    if action == 0:  # Move Up
        row = max(row - 1, 0)
    elif action == 1:  # Move Down
        row = min(row + 1, n_rows - 1)
    elif action == 2:  # Move Left
        col = max(col - 1, 0)
    elif action == 3:  # Move Right
        col = min(col + 1, n_cols - 1)

    return row, col  

max_episodes = 1000

for episode in range(max_episodes):
    episode_states = [] 
    state = 0  

    while True:
        episode_states.append(state)
        action = random.randint(0, n_actions - 1) 
        next_row, next_col = calculate_next_state(state, action)
        next_state = next_row * n_cols + next_col

        if next_state == 8: 
            break

        state = next_state

    # Calculate returns and update state values
    G = rewards[next_state]

    for i, s in enumerate(reversed(episode_states)):
        if i == 0:
            V[s] = V[s] + alpha * (G - V[s])
        else:
            G = gamma * G + rewards[s]
            V[s] = V[s] + alpha * (G - V[s])

state_values_matrix = V.reshape(n_rows, n_cols)
print(state_values_matrix)

csv_filename_matrix = "state_values_matrix.csv"
with open(csv_filename_matrix, mode='w', newline='') as state_values_matrix_file:
    state_values_matrix_writer = csv.writer(state_values_matrix_file)
    
    for row in state_values_matrix:
        state_values_matrix_writer.writerow(row)

