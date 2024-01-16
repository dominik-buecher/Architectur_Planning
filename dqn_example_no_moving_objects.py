import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchinfo import summary
import random
from collections import namedtuple
import numpy as np

print("GPU support: ", torch.cuda.is_available())

# Define the environment
n_rows = 10
n_cols = 10
# n_states = n_rows * n_cols
n_actions = 4 
actions = ['Up', 'Down', 'Left', 'Right']

goal_field = (n_rows-1, n_cols-1)
goal_reward = 100

# Define the action-to-index dictionary
index_to_action = {0: 'Up', 1: 'Down', 2:'Left', 3: 'Right'}

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x.view(-1, self.fc1.in_features)))
        return self.fc2(x)

# Define rewards and penalties
rewards = np.zeros((n_rows, n_cols))
rewards[goal_field[0]][goal_field[1]] = goal_reward
rewards[5][5]   = 9
rewards[6][6]   = 9

# fields_travelled = []
def get_reward_for_field(row_idx, col_idx):
    if rewards[row_idx][col_idx] == 0:
        return -1
    else:
        return rewards[row_idx][col_idx]

# DQN parameters
gamma = 0.6          # Discount factor
epsilon = 1.0      # Initial exploration rate
epsilon_decay = 0.9997
min_epsilon = 0.1
alpha = 0.20
batch_size = 32
target_update = 100
max_episodes = 4000

# Initialize replay memory
replay_memory_size = 4000
replay_memory = []

# Define named tuple for experience replay
Transition = namedtuple('Transition', ('state_idx', 'action', 'next_state_idx', 'reward', 'done'))

# Initialize DQN and target DQN
input_size = 1  # Only the state index
output_size = n_actions
dqn = DQN(input_size, output_size)
target_dqn = DQN(input_size, output_size)
target_dqn.load_state_dict(dqn.state_dict())
target_dqn.eval()
summary(target_dqn)

# Define optimizer and loss function
optimizer = optim.Adam(dqn.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

def calculate_next_state(current_state, action):
    # Define the transition rules based on the chosen action
    row, col = current_state // n_cols, current_state % n_cols

    if action == 0:  # Move Up
        row = row - 1
        row = max(0, row)
    elif action == 1:  # Move Down
        row = row + 1
        row = min(row, n_rows-1)
    elif action == 2:  # Move Left
        col = col - 1
        col = max(0,col)
    elif action == 3:  # Move Right
        col = col + 1
        col = min(col, n_cols-1)

    return row, col 

# # Function to select epsilon-greedy action
# def epsilon_greedy_action(state, epsilon):
#     if random.uniform(0, 1) < epsilon:
#         while True:
#             action = random.randint(0, n_actions - 1)
#             next_row, next_col = calculate_next_state(state, action)
#             if not (next_row < 0 or next_row >= n_rows or next_col < 0 or next_col >= n_cols):
#                 break
#     else:
#         state_tensor = torch.tensor([state], dtype=torch.float32)  # Wrap the state in a tensor
#         q_values = dqn(state_tensor)
#         max_action = q_values.argmax().item()
#         # print("q_values.shape = ", q_values.shape)
#         # print("q_values", q_values)
#         # print(max_action)
#         assert(max_action >= 0 and max_action <= 3)

#         next_row, next_col = calculate_next_state(state, max_action)
#         if next_row < 0 or next_row >= n_rows or next_col < 0 or next_col >= n_cols:
#             while True:
#                 action = random.randint(0, n_actions - 1)
#                 next_row, next_col = calculate_next_state(state, action)
#                 if not (next_row < 0 or next_row >= n_rows or next_col < 0 or next_col >= n_cols):
#                     break
#         else:
#             action = max_action
#     return action

def is_allowed_action(next_row, next_col):
    if next_row >= 0 and next_row < n_rows and next_col >= 0 and next_col < n_cols:
        return True
    else:
        return False

# Function to select epsilon-greedy action
def epsilon_greedy_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        # Take random action
        while True:
            action = random.randint(0, n_actions - 1)
            next_row, next_col = calculate_next_state(state, action)
            if is_allowed_action(next_row, next_col):
                break
    else:
        # Take action proposed by DQN
        state_tensor = torch.tensor([state], dtype=torch.float32)  # Wrap the state in a tensor
        q_values = dqn(state_tensor)
        max_action = q_values.argmax().item()
        # print("q_values.shape = ", q_values.shape)
        # print("q_values", q_values)
        # print(max_action)
        assert(max_action >= 0 and max_action <= 3)

        next_row, next_col = calculate_next_state(state, max_action)

        # If action is not allowed, anyways take allowed random action
        if not is_allowed_action(next_row, next_col):
            while True:
                action = random.randint(0, n_actions - 1)
                next_row, next_col = calculate_next_state(state, action)
                if is_allowed_action(next_row, next_col):
                    break
        else:
            action = max_action
    return action

goal_reached_counter = 0
def is_goal_reached(reward):
    global goal_reached_counter
    if reward == goal_reward:
        goal_reached_counter += 1
        # print("goal reached!!")
        return True
    else:
        return False

action_counter = [0,0,0,0]

# DQN training loop
for episode_idx in range(max_episodes):
    row = 0
    col = 0
    total_reward = 0
    done = False
    step_count = 0

    if (episode_idx % 1000) == 0:
        print("Number of trained episodes: ", episode_idx)
        print("Current epsilon =", epsilon)
        print("Goal reached:", goal_reached_counter, "times")

    while not done:
        state_idx = row * n_cols + col
        # print("state_idx = ", state_idx)
        action = epsilon_greedy_action(state_idx, epsilon)
        action_counter[action] += 1
        #print(state_idx, index_to_action[action])
        
        # Simulate environment (transition to next state and get reward)
        next_row, next_col = calculate_next_state(state_idx, action)
        next_state_idx = next_row * n_cols + next_col
        
        # reward = rewards[next_row][next_col]
        reward = get_reward_for_field(next_row, next_col)
        done = is_goal_reached(reward)                    # Check if goal reached
        total_reward += reward
        
        # Store transition in replay memory
        transition = Transition(state_idx, action, next_state_idx, reward, done)
        replay_memory.append(transition)
        if len(replay_memory) > replay_memory_size:
            replay_memory.pop(0)  # Remove oldest transition if memory is full

        # Sample a random batch from replay memory
        if len(replay_memory) > batch_size:
            transitions = random.sample(replay_memory, batch_size)
            state_batch = torch.tensor([t.state_idx for t in transitions], dtype=torch.float32)
            action_batch = torch.tensor([t.action for t in transitions], dtype=torch.int64).unsqueeze(1)
            reward_batch = torch.tensor([t.reward for t in transitions], dtype=torch.float32)
            next_state_batch = torch.tensor([t.next_state_idx for t in transitions], dtype=torch.float32)
            done_batch = torch.tensor([t.done for t in transitions], dtype=torch.float32)
            # print("state_batch = ", state_batch.shape)
            # print("action_batch = ", action_batch.shape)
            # print("reward_batch = ", reward_batch.shape)
            # print("next_state_batch = ", next_state_batch.shape)

            # Calculate Q-values for current and next states
            q_values = dqn(state_batch)
            # print("action_batch: ", action_batch)
            q_values = q_values.gather(1, action_batch)
            next_q_values = target_dqn(next_state_batch).max(1)[0].detach()
            # expected_q_values = reward_batch + gamma * (1 - done_batch) * next_q_values

            
            # print("reward_batch.shape      => ", reward_batch.shape          )
            # print("done_batch.shape        => ", done_batch.shape            )
            # print("next_q_values.shape     => ", next_q_values.shape         )
            expected_q_values = reward_batch + gamma * (1 - done_batch) * next_q_values
            # print("expected_q_values.shape => ", expected_q_values.shape     )

            # Compute loss and update DQN
            loss = loss_fn(q_values, expected_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #print(state, next_state_idx, action, step_count+1)

        row = next_row
        col = next_col
        step_count += 1

        if step_count > 40:  # Terminate the episode if steps exceed 20
            done = True

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if episode_idx % target_update == 0:
        target_dqn.load_state_dict(dqn.state_dict())  # Update target network

print("                                 ", actions)
print("Actions performed during training", action_counter)

# After training, generate actions for an episode
generated_actions = []
state = 0  # Starting state
done = False
step_count = 0

while not done:
    q_values = dqn(torch.tensor([state], dtype=torch.float32))
    action = q_values.argmax().item()
    next_row, next_col = calculate_next_state(state, action)
    action = torch.topk(q_values, 2).indices[0, 1].item() if next_row < 0 or next_row >= n_rows or next_col < 0 or next_col >= n_cols else action
    generated_actions.append(index_to_action[action])
    print(int(state/n_rows), state%n_cols, index_to_action[action])
    print("Next Position:", next_row, next_col)

    next_row, next_col = calculate_next_state(state, action)
    next_state_idx = next_row * n_cols + next_col
    
    done = True if (next_row == goal_field[0] and next_col == goal_field[1]) else False  # Check if goal reached

    state = next_state_idx
    step_count += 1
    done = True if step_count > 50 else done

print("Generated Actions:", generated_actions)



