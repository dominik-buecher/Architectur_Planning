import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import namedtuple
import numpy as np
import tkinter as tk
from window import *

# Define the environment
n_rows = 3
n_cols = 3
n_states = n_rows * n_cols
n_actions = 4 
actions = ['Up', 'Down', 'Left', 'Right']

# Define the action-to-index dictionary
index_to_action = {0: 'Up', 1: 'Down', 2:'Left', 3: 'Right'}

root = tk.Tk()
root.title("Grid Window with Cows, Mower, and Target")

rows = 20
cols = 20
num_cows = 5

grid_window = GridWindow(root, rows, cols, num_cows)


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x.view(-1, self.fc1.in_features)))
        return self.fc2(x)

# Define rewards and penalties
rewards = np.zeros(n_states)
rewards[2] = -1
rewards[4] = -1
rewards[8] = 1  # Goal state

# DQN parameters
gamma = 1  # Discount factor
epsilon = 1.0  # Initial exploration rate
epsilon_decay = 0.9
min_epsilon = 0.1
alpha = 0.1
batch_size = 32
target_update = 10
max_episodes = 100

# Initialize replay memory
replay_memory_size = 10000
replay_memory = []

# Define named tuple for experience replay
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

# Initialize DQN and target DQN
input_size = 1  # Only the state index
output_size = n_actions
dqn = DQN(input_size, output_size)
target_dqn = DQN(input_size, output_size)
target_dqn.load_state_dict(dqn.state_dict())
target_dqn.eval()

# Define optimizer and loss function
optimizer = optim.Adam(dqn.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

def calculate_next_state(action):
    
    state = grid_window.get_state()
    row, col = grid_window.get_future_state(state, action)

    return row, col 

# Function to select epsilon-greedy action
def epsilon_greedy_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        while True:
            action = random.randint(0, n_actions - 1)
            next_row, next_col = calculate_next_state(action)
            if not (next_row < 0 or next_row >= n_rows or next_col < 0 or next_col >= n_cols):
                break
    else:
        state_tensor = torch.tensor([state], dtype=torch.float32)  # Wrap the state in a tensor
        q_values = dqn(state_tensor)
        max_action = q_values.argmax().item()
        next_row, next_col = calculate_next_state(max_action)
        if next_row < 0 or next_row >= n_rows or next_col < 0 or next_col >= n_cols:
            while True:
                action = random.randint(0, n_actions - 1)
                next_row, next_col = calculate_next_state(action)
                if not (next_row < 0 or next_row >= n_rows or next_col < 0 or next_col >= n_cols):
                    break
        else:
            action = max_action
    return action

# DQN training loop
for episode in range(max_episodes):
    row = 0
    col = 0
    total_reward = 0
    done = False
    step_count = 0

    while not done:
        state = grid_window.get_state()
        action = epsilon_greedy_action(state, epsilon)
        
        #print(state, index_to_action[action])
        
        # Simulate environment (transition to next state and get reward)
        next_row, next_col = calculate_next_state(action)
        next_state = next_row * n_cols + next_col
        reward = grid_window.get_reward(state, next_row, next_col)
        done = True if reward == 50 else False  # Check if goal reached
        total_reward += reward
        
        # Store transition in replay memory
        transition = Transition(state, action, next_state, reward, done)
        replay_memory.append(transition)
        if len(replay_memory) > replay_memory_size:
            replay_memory.pop(0)  # Remove oldest transition if memory is full

        # Sample a random batch from replay memory
        if len(replay_memory) > batch_size:
            transitions = random.sample(replay_memory, batch_size)
            state_batch = torch.tensor([t.state for t in transitions], dtype=torch.float32)
            action_batch = torch.tensor([t.action for t in transitions], dtype=torch.int64).unsqueeze(1)
            reward_batch = torch.tensor([t.reward for t in transitions], dtype=torch.float32)
            next_state_batch = torch.tensor([t.next_state for t in transitions], dtype=torch.float32)
            done_batch = torch.tensor([t.done for t in transitions], dtype=torch.float32)

            # Calculate Q-values for current and next states
            q_values = dqn(state_batch)
            print("action_batch: ", action_batch)
            q_values = q_values.gather(1, action_batch)
            next_q_values = target_dqn(next_state_batch).max(1)[0].detach()
            expected_q_values = reward_batch + gamma * (1 - done_batch) * next_q_values

            # Compute loss and update DQN
            loss = loss_fn(q_values, expected_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #print(state, next_state, action, step_count+1)

        row = next_row
        col = next_col
        step_count += 1

        if step_count > 20:  # Terminate the episode if steps exceed 20
            done = True

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if episode % target_update == 0:
        target_dqn.load_state_dict(dqn.state_dict())  # Update target network


# After training, generate actions for an episode
generated_actions = []
state = 0  # Starting state
done = False
step_count = 0

while not done:
    q_values = dqn(torch.tensor([state], dtype=torch.float32))
    action = q_values.argmax().item()
    next_row, next_col = calculate_next_state(action)
    action = torch.topk(q_values, 2).indices[0, 1].item() if next_row < 0 or next_row >= n_rows or next_col < 0 or next_col >= n_cols else action
    generated_actions.append(index_to_action[action])
    print(state, index_to_action[action])

    next_row, next_col = calculate_next_state(action)
    next_state = next_row * n_cols + next_col
    
    done = True if next_state == 8 else False  # Check if goal reached

    state = next_state

print("Generated Actions:", generated_actions)



