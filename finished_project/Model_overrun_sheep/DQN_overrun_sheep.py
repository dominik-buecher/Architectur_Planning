import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as FF
import random
from collections import namedtuple
import tkinter as tk
import keyboard
import json
from environment2 import *



###################################################################################
# PLEAS READ THE README FILE! 
# We got new changes which we discribe in the README File
###################################################################################




LOSS = []
rows = 15
cols = 15
n_states = rows * cols
n_actions = 5
num_cows = 5
input_channels = 2



output_size = n_actions
root = tk.Tk()
root.title("Grid Window with Sheeps, Mower, and Target")
# Creating the environment
grid_window = GridWindow(root, rows, cols, num_cows)


class DQN(nn.Module):
    def __init__(self, input_channels, output_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)  # Adjusting the size of the hidden layer
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # Change the order of dimensions
        x = FF.relu(self.conv1(x))
        x = FF.max_pool2d(x, 2)
        x = FF.relu(self.conv2(x))
        x = FF.max_pool2d(x, 2)
        x = x.reshape(x.size(0), -1)  # Flattening
        x = FF.relu(self.fc1(x))
        return self.fc2(x)



# Experience is stored as Transition -> needed for Memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

# ReplayMemory is used so the agent can learn from past experiences
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
    def __init__(self, input_channels, output_size, capacity=30000, batch_size=32, gamma=0.999, epsilon=1.0, epsilon_decay=0.999, min_epsilon=0.1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_channels = input_channels
        self.output_size = output_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size

        self.policy_net = DQN(input_channels, output_size).to(self.device)
        self.target_net = DQN(input_channels, output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = ReplayMemory(capacity)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.output_size - 1)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state.unsqueeze(0).to(self.device))
                return q_values.argmax().item()
            
    def select_action_netz(self, state):
        q_values = self.policy_net(state[0].unsqueeze(0).to(self.device))
        return q_values.argmax().item()


    def train(self):
        if len(self.memory.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Use batch.state directly without conversion
        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.int64).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_state_batch = torch.stack(batch.next_state).to(self.device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1).to(self.device)


        q_values = self.policy_net(state_batch).gather(1, action_batch)
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()

        expected_q_values = reward_batch + self.gamma * (1 - done_batch) * next_q_values
        expected_q_values = expected_q_values.max(dim=1, keepdim=True)[0]
        loss = FF.mse_loss(q_values, expected_q_values)
        LOSS.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)



def test_model(cows_pos):
    """
    Performs a test run of the model where actions are selected 100% by the neural network.
    This allows to check how successful the training was.

    """
    grid_window.reset_to_initial_state(cows_pos)
    # generated_actions = []
    state = grid_window.get_state()
    done = False
    total_reward = 0
    action_counter = 0
    while not done:
        # Choose action
        action = dqn_agent.select_action_netz(state)
        action_counter += 1
        
        # Choose new state based on action
        next_row, next_col = grid_window.get_future_state(action)
        reward = grid_window.get_reward(next_row, next_col, action)
        grid_window.move_mower_abs(next_row, next_col)
        # Cows are moved only every fifth step
        if action_counter % 5 == 0:
            grid_window.move_cows()
        grid_window.root.update()
        grid_window.root.after(150)
        print("action: ", action)
        print("reward: ", reward)
        
        
        next_state = grid_window.get_state() 
        total_reward += reward

        done = True if next_row == grid_window.target.grid_info()["row"] and next_col == grid_window.target.grid_info()["column"] or action_counter > 600 else False
        state = next_state



# Create the DQN agent (an already trained model can be further trained optionally)
dqn_agent = DQNAgent(input_channels, output_size)
# loaded_state_dict = torch.load('dqn_5_actions/models/dqn_model_onehot_nocow_99percent.pth')
# dqn_agent.policy_net.load_state_dict(loaded_state_dict)


######################################################
# Specify whether a model should be tested or a new model should be trained
#train = True
train = False
######################################################

# Training loop
if train:
    initial_state, cows_pos = grid_window.get_state()
    max_episodes = 500
    max_actions = 5000
    i = 1
    for episode in range(max_episodes):
        print("episode: ", episode)
        state = initial_state
        grid_window.reset_to_initial_state(cows_pos)
        total_reward = 0
        done = False
        action_counter = 0
        consecutive_unvisited_count = 0
        while not done:
            action = dqn_agent.select_action(state)
            action_counter += 1
            
            next_row, next_col = grid_window.get_future_state(action)
            reward = grid_window.get_reward(next_row, next_col, action)

            grid_window.move_mower_abs(next_row, next_col)
            # Cows are moved only every fifth step
            if action_counter % 5 == 0:
                grid_window.move_cows()
            grid_window.root.update()
            grid_window.root.after(1)

            next_state, _ = grid_window.get_state()  # Update the state
            done = True if next_row == grid_window.target.grid_info()["row"] and next_col == grid_window.target.grid_info()["column"] else False

            dqn_agent.memory.push(state, action, next_state, reward, done)
            dqn_agent.train()

            total_reward += reward
            state = next_state

        dqn_agent.decay_epsilon()

        # The target network is updated every fifth episode, and the current model is tested and saved
        if episode % 5 == 0:
            dqn_agent.update_target_net()
            print("Testing model...")
            test_model(cows_pos)
            print("Testing model finished!")
            print("Saving model...")
            torch.save(dqn_agent.policy_net.state_dict(), 'Model_overrun_sheep/dqn_model_5cow_overrun_sheep_1.pth')
            print("Model saved!")

        # If 'q' is pressed while an epoch is finished, the current model is saved
        if keyboard.is_pressed('q'):
            print("Saving model...")
            torch.save(dqn_agent.policy_net.state_dict(), 'Model_overrun_sheep/dqn_model_5cow_overrun_sheep_1.pth')
            print("Model saved!")

            # Save the list as JSON
            with open('loss_werte_nocow.json', 'w') as f:
                json.dump(LOSS, f)


# Test the selected model
dqn_agent = DQNAgent(input_channels, output_size)
loaded_state_dict = torch.load('Model_overrun_sheep/dqn_model_5cow_overrun_sheep.pth')
dqn_agent.policy_net.load_state_dict(loaded_state_dict)
initial_state, cows_pos = grid_window.get_state()
test_model(cows_pos)


