import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as FF
from torchinfo import summary
import random
from collections import namedtuple
import tkinter as tk
import keyboard
import math
import json
from window_onehot2 import *

LOSS = []
rows = 15
cols = 15
n_states = rows * cols
n_actions = 5 
num_cows = 0
input_channels = 2  # Zustand + Positionen der Kühe
output_size = n_actions
length_state = 4 + (2 * num_cows)
root = tk.Tk()
root.title("Grid Window with Cows, Mower, and Target")
# Erstellen des Enviroments
grid_window = GridWindow(root, rows, cols, num_cows)


class DQN(nn.Module):
    def __init__(self, input_channels, output_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)  # Anpassen der Größe der versteckten Schicht
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # Ändert die Reihenfolge der Dimensionen
        x = FF.relu(self.conv1(x))
        x = FF.max_pool2d(x, 2)
        x = FF.relu(self.conv2(x))
        x = FF.max_pool2d(x, 2)
        x = x.reshape(x.size(0), -1)  # Flattening
        x = FF.relu(self.fc1(x))
        return self.fc2(x)



# Erfahrung wird als Transition abgepeichert -> wird für Memory gebraucht
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

# ReplayMemory wird verwendet damit der Agent aus alten Erfahrungen lernen kann
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

# Definiere den DQN agent
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
                #state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)  # Zustand in den richtigen Tensor umwandeln
                # Annahme: state ist bereits in der richtigen Form (z.B. [batch_size, input_channels, height, width])
                #state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
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

        # Nutze batch.state direkt ohne Konvertierung
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
    grid_window.reset_to_initial_state(cows_pos)
    #generated_actions = []
    state = grid_window.get_state()
    done = False
    total_reward = 0
    action_counter = 0
    while not done:
        # Aktion wählen
        action = dqn_agent.select_action_netz(state)
        action_counter += 1
        #print("action; ", action)
        
        # Neuen Zustand wählen basierend auf der Aktion
        next_row, next_col = grid_window.get_future_state(action)
        reward = grid_window.get_reward(next_row, next_col, action)

        grid_window.move_mower_abs(next_row, next_col)
        grid_window.move_cows()
        grid_window.root.update()
        grid_window.root.after(50)
        
        next_state = grid_window.get_state()
        
        total_reward += reward

        done = True if next_row == grid_window.target.grid_info()["row"] and next_col == grid_window.target.grid_info()["column"] else False
        print("reward: ", reward)
        #print("total_reward: ", total_reward)
        state = next_state



n_actions = 5
index_to_action = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right', 4: 'Stay'}

# Create the DQN agent
dqn_agent = DQNAgent(input_channels, output_size)
# loaded_state_dict = torch.load('dqn_5_actions/models/dqn_model_onehot_nocow_99percent.pth')
# dqn_agent.policy_net.load_state_dict(loaded_state_dict)


#train = True
train = False

if train:
    initial_state, cows_pos = grid_window.get_state()
    max_episodes = 50000
    max_actions = 9000
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
            if action_counter % 5 == 0:
                grid_window.move_cows()
            grid_window.root.update()
            grid_window.root.after(1)

            next_state, _ = grid_window.get_state()  # Update the state

            
            # if grid_window.cells[next_row][next_col]["bg"] == "#006400":
            #     done = True 
            #     reward -= 50

            done = True if next_row == grid_window.target.grid_info()["row"] and next_col == grid_window.target.grid_info()["column"] else False

            dqn_agent.memory.push(state, action, next_state, reward, done)
            dqn_agent.train()

            total_reward += reward
         
            state = next_state

        dqn_agent.decay_epsilon()
        print("action_counter: ", action_counter)
        print("total_reward: ", total_reward)

        if episode % 5 == 0:
            dqn_agent.update_target_net()
            print("Testing model...")
            test_model(cows_pos)
            print("Testing model finished!")
            torch.save(dqn_agent.policy_net.state_dict(), r'C:\Users\domin\Documents\Studium\Master\Semester_2\Achritecture_and_Planning\Architectur_Planning\dqn_5_actions\models\dqn_model_onehot_2cow-1.pth')
            

        if keyboard.is_pressed('q'):
            print("Saving model...")
            torch.save(dqn_agent.policy_net.state_dict(), r'C:\Users\domin\Documents\Studium\Master\Semester_2\Achritecture_and_Planning\Architectur_Planning\dqn_5_actions\models\dqn_model_onehot_2cow-1.pth')
            print("Model saved!")

            # Speichere die Liste als JSON
            with open('loss_werte_nocow.json', 'w') as f:
                json.dump(LOSS, f)
    #torch.save(dqn_agent.policy_net.state_dict(), 'models/dqn_model_standart_cow0_1.pth')


dqn_agent = DQNAgent(input_channels, output_size)
loaded_state_dict = torch.load(r'C:\Users\domin\Documents\Studium\Master\Semester_2\Achritecture_and_Planning\Architectur_Planning\dqn_5_actions\models\dqn_model_onehot_nocow_finished_perfect.pth')
dqn_agent.policy_net.load_state_dict(loaded_state_dict)
initial_state, cows_pos = grid_window.get_state()
test_model(cows_pos)


