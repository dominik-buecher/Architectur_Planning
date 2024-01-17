import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import namedtuple
import tkinter as tk
from window import *


n_rows = 20
n_cols = 20
n_states = n_rows * n_cols
n_actions = 4 

rows = 20
cols = 20
num_cows = 5

root = tk.Tk()
root.title("Grid Window with Cows, Mower, and Target")

grid_window = GridWindow(root, rows, cols, num_cows)


# Define the DQN model
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Define the experience replay memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

# Define the DQN agent
class DQNAgent:
    def __init__(self, input_size, output_size, capacity=10000, batch_size=32, gamma=0.9, epsilon=1.0, epsilon_decay=0.99, min_epsilon=0.1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.output_size = output_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size

        self.policy_net = DQN(input_size, output_size).to(self.device)
        self.target_net = DQN(input_size, output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = ReplayMemory(capacity)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.output_size - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor([state], dtype=torch.float32).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def train(self):
        if len(self.memory.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.tensor(batch.state, dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.int64).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(self.device)
        next_state_batch = torch.tensor(batch.next_state, dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32).to(self.device)

        q_values = self.policy_net(state_batch).gather(1, action_batch)
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()

        expected_q_values = reward_batch + self.gamma * (1 - done_batch) * next_q_values

        loss = F.mse_loss(q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        #print("loss.backward() start!")
        loss.backward()
        #print("loss.backward() finished!")
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

# Define the environment-specific parameters
n_rows = 20
n_cols = 20
n_states = n_rows * n_cols
n_actions = 4
index_to_action = {0: 'Up', 1: 'Down', 2:'Left', 3: 'Right'}


# Create the DQN agent
input_size = 414
output_size = n_actions
dqn_agent = DQNAgent(input_size, output_size)

#train = True
train = False

if train is True:
    # Training loop
    max_episodes = 1000
    for episode in range(max_episodes):
        state = grid_window.get_state()
        total_reward = 0
        done = False
        print("episode: ", episode)
        action_counter = 0
        while not done:
            action = dqn_agent.select_action(state)
            action_counter += 1
            next_row, next_col = grid_window.get_future_state(state, action)
            next_state = grid_window.get_state()  # Update the state
            reward = grid_window.get_reward(state, next_row, next_col)
            done = True if reward == 500 or action_counter >= 1000  else False
            #print("reward: ", reward)
            dqn_agent.memory.push(state, action, next_state, reward, done)
            dqn_agent.train()

            state = next_state
            total_reward += reward

        dqn_agent.decay_epsilon()
        if episode % 10 == 0:
            dqn_agent.update_target_net()

    torch.save(dqn_agent.policy_net.state_dict(), 'models/dqn_model.pth')

else:
    loaded_state_dict = torch.load('models/dqn_model.pth')
    dqn_agent.policy_net.load_state_dict(loaded_state_dict)



# After training, use the learned policy
generated_actions = []
state = grid_window.get_state()
done = False
# Save the DQN model

total_reward = 0
while not done:
    action = dqn_agent.select_action(state)
    next_row, next_col = grid_window.get_future_state(state, action)
    next_state = grid_window.get_state()  # Update the state
    generated_actions.append(index_to_action[action])
    state = next_state
    reward = grid_window.get_reward(state, next_row, next_col)
    total_reward += reward
    done = True if state[-1] == 1 or total_reward > 500 else False  # Check if goal reached
    print("total_reward: ", total_reward)
    
print("Generated Actions:", generated_actions)
