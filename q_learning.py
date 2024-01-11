import numpy as np
import random

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

# Q-learning training parameters
gamma = 0.9  
epsilon = 1.0  
epsilon_decay = 0.995
min_epsilon = 0.1
max_episodes = 1000
alpha = 0.9

# Q-table initialization
q_table = np.zeros((n_states, n_actions)) 

def calculate_next_state(current_state, action):
    row, col = current_state // n_cols, current_state % n_cols

    if action == 0:  # Move Up
        row = row - 1
    elif action == 1:  # Move Down
        row = row + 1
    elif action == 2:  # Move Left
        col = col -1
    elif action == 3:  # Move Right
        col = col + 1

    return row, col  

def epsilon_greedy_action(q_values, epsilon, n_actions, state):
    if random.uniform(0, 1) < epsilon:
        while True:
            action = random.randint(0, n_actions - 1)
            next_row, next_col = calculate_next_state(state, action)

            if not (next_row < 0 or next_row >= n_rows or next_col < 0 or next_col >= n_cols):
                break
    else:
        max_action = np.argmax(q_values)
        
        next_row, next_col = calculate_next_state(state, max_action)

        if next_row < 0 or next_row >= n_rows or next_col < 0 or next_col >= n_cols:
            while True:
                action = random.randint(0, n_actions - 1)
                next_row, next_col = calculate_next_state(state, action)

                if not (next_row < 0 or next_row >= n_rows or next_col < 0 or next_col >= n_cols):
                    break
        else:
            action = max_action

    return action

# Q-learning training loop
for episode in range(max_episodes):
    row = 0
    col = 0
    done = False
    step_count = 0  

    while not done:
        state_index = row * n_cols + col
        q_values = q_table[state_index, :]
        action = epsilon_greedy_action(q_values, epsilon, n_actions, state_index)

        next_row, next_col = calculate_next_state(state_index, action)
        next_state_index = next_row * n_cols + next_col
        reward = rewards[next_state_index]
        #print(state_index, action, next_state_index, epsilon)

        # Q-learning update
        q_target = reward + gamma * np.max(q_table[next_state_index])
        q_table[state_index][action] = (1 - alpha) * q_table[state_index][action] + alpha * q_target

        if state_index == 8:
            #print("Goal reached")
            done = True

        if step_count > 10:  # Terminate the episode if steps exceed 10
            done = True

        if epsilon > min_epsilon:
            epsilon *= epsilon_decay

        step_count += 1

        row = next_row
        col = next_col

# Print the Q-Table
print(q_table)
print(q_table.shape)


import csv
csv_filename = "q_table.csv"
with open(csv_filename, mode='w', newline='') as q_table_file:
    q_table_writer = csv.writer(q_table_file)
    header = ['State'] + [f'Action_{i}' for i in range(n_actions)]
    q_table_writer.writerow(header)

    # Write each state and its associated Q-values
    for state_index, q_values in enumerate(q_table):
        row = [state_index] + q_values.tolist()
        q_table_writer.writerow(row)


