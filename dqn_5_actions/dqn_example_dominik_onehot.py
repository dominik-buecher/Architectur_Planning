import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import namedtuple
import tkinter as tk
import keyboard
import math
from window_onehot import *

rows = 20
cols = 20
n_states = rows * cols
n_actions = 5 
num_cows = 0
input_size = (n_states + 4 + (2 * num_cows))  # Zustand + Positionen der Kühe
output_size = n_actions
coordinates = 4 + (2 * num_cows)
root = tk.Tk()
root.title("Grid Window with Cows, Mower, and Target")
# Erstellen des Enviroments
grid_window = GridWindow(root, rows, cols, num_cows)


# DQN model
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
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
    def __init__(self, input_size, output_size, capacity=30000, batch_size=32, gamma=0.999, epsilon=1.0, epsilon_decay=0.999, min_epsilon=0.1):
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
                state_tensor = F.one_hot(torch.tensor([state]), num_classes=rows).float().to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()


    def train(self):
        if len(self.memory.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)

        batch = Transition(*zip(*transitions))
        state_batch = F.one_hot(torch.tensor(batch.state), num_classes=rows).float().to(self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.int64).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(self.device)
        next_state_batch = F.one_hot(torch.tensor(batch.next_state), num_classes=rows).float().to(self.device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32).to(self.device)

        q_values = self.policy_net(state_batch).gather(1, action_batch)
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()

        expected_q_values = reward_batch + self.gamma * (1 - done_batch) * next_q_values

        loss = F.mse_loss(q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


n_actions = 5
index_to_action = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right', 4: 'Stay'}

# Create the DQN agent
dqn_agent = DQNAgent(n_states, output_size)
# loaded_state_dict = torch.load('models/dqn_model_standart_cow0_1.pth')
# dqn_agent.policy_net.load_state_dict(loaded_state_dict)


train = True
#train = False

if train:
    initial_state = grid_window.get_state(coordinates)
    print("initial_state: ", initial_state)
    max_episodes = 50000
    max_actions = 9000
    for episode in range(max_episodes):
        # aktuellen state laden
        state = initial_state
        grid_window.reset_to_initial_state(initial_state)
        total_reward = 0
        done = False
        print("episode: ", episode)
        action_counter = 0
        consecutive_unvisited_count = 0
        # Schleife durchlaufen bis bedingung erfüllt ist
        while not done:
            # Wähle Aktion basierend auf aktuellem Zustand
            action = dqn_agent.select_action(state)
            action_counter += 1
            
            # Erhalte zukünfitgen State welcher durch die Aktion erreicht wird
            next_row, next_col = grid_window.get_future_state(state, action)

            grid_window.move_mower_abs(next_row, next_col, coordinates)
            grid_window.move_cows()
            grid_window.root.update()
            grid_window.root.after(1)

            next_state = grid_window.get_state(coordinates)  # Update the state
            
            # One-Hot-Kodierung des Zustands
            state_one_hot = F.one_hot(torch.tensor([state]), num_classes=rows).float().to(dqn_agent.device)
            next_state_one_hot = F.one_hot(torch.tensor([next_state]), num_classes=rows).float().to(dqn_agent.device)

            # Berechne Reward basierend auf dem zukünftigen Zustand 
            reward, consecutive_unvisited_count = grid_window.get_reward(state, next_row, next_col, action, action_counter, num_cows, consecutive_unvisited_count)
            
            # Wenn Reward = 500 erreicht ist oder 1000 Aktionen durchgeführt wurden -> Schleife abbrechen
            #done = True if reward >= 5000 else False
            done = True if grid_window.is_90_percent_visited(state, 20, 20) is True and next_row == grid_window.target.grid_info()["row"] and next_col == grid_window.target.grid_info()["column"] else False
            visited = grid_window.is_single_field_visited(state, next_row, next_col, 20, 20)
            
            if not (state[0] == next_row and state[1] == next_col):
                if visited is True:
                    done = True

            if action_counter >= max_actions:
                reward += -30
                done = True
            
            # Fügt den durchlauf als Transition in die Memory hinzu
            dqn_agent.memory.push(state_one_hot, action, next_state_one_hot, reward, done)
            dqn_agent.train()

            total_reward += reward
         
            #print("action: ", action)
            state = next_state

        max_actions = max(4000, max_actions - 50)
        # Verringern der Wahrscheinlichkeit für zufällige Aktionen
        dqn_agent.decay_epsilon()
        print("action_counter: ", action_counter)
        print("total_reward: ", total_reward)
        # Nach 10 Episoden wird das Netz geupdated
        if episode % 5 == 0:
            dqn_agent.update_target_net()

        if keyboard.is_pressed('q'):
            print("Saving model...")
            torch.save(dqn_agent.policy_net.state_dict(), 'models/dqn_model_standart_cow0_1.pth')
            print("Model saved!")

    # trainiertes Modell speichern
    torch.save(dqn_agent.policy_net.state_dict(), 'models/dqn_model_standart_cow0_1.pth')


# Wird verwendet um die SChritte des Agent in dem Enviorment darzustellen
def play_environment(grid_window, actions, index_to_action):
    for action in actions:
        print("action: ", action)
        
        if action in index_to_action.values():
            # prüfen ob Aktion im Dictionary enthalten ist
            action_index = next((index for index, act in index_to_action.items() if act == action), None)
            
            # Wähle neuen Zustand basierend auf der Aktion
            future_row, future_col = grid_window.get_future_state(grid_window.get_state(coordinates), action_index)
            
            # Bewege Rasenmäher zu neuem Zustand
            grid_window.move_mower_abs(future_row, future_col, coordinates)
        
        grid_window.move_cows()
        grid_window.root.update()
        grid_window.root.after(500)

        # Überprüfe, ob alle Felder besucht und Ziel erreicht
        if grid_window.get_state(coordinates)[-1] == 1 and grid_window.get_state(coordinates)[-2] == 1:
            print("Spiel beendet! Ziel erreicht und alle Felder besucht.")
            break




#generated_actions = []
state = grid_window.get_state(coordinates)
done = False
total_reward = 0
action_counter = 0
consecutive_unvisited_count = 0
while not done:
    # Aktion wählen
    action = dqn_agent.select_action_Netz(state)
    action_counter += 1
    print("action; ", action)
    
    # Neuen Zustand wählen basierend auf der Aktion
    next_row, next_col = grid_window.get_future_state(state, action)

    grid_window.move_mower_abs(next_row, next_col, coordinates)
    grid_window.move_cows()
    grid_window.root.update()
    grid_window.root.after(100)
    
    next_state = grid_window.get_state(coordinates)
    
    # Alle Aktionen abspeichern
    #generated_actions.append(index_to_action[action])

    
    # Reward berechnen basierend auf der Aktino
    reward, consecutive_unvisited_count = grid_window.get_reward(state, next_row, next_col, action, action_counter, num_cows, consecutive_unvisited_count)
    total_reward += reward

    
    # Abbruchbedingung prüfen: Ziel erreicht? oder Reward über 100
    
    done = True if grid_window.is_90_percent_visited(state, 20, 20) is True and next_row == grid_window.target.grid_info()["row"] and next_col == grid_window.target.grid_info()["column"] else False
    print("total_reward: ", total_reward)
    state = next_state
    
# print("Generated Actions:", generated_actions)

# play_environment(grid_window, generated_actions, index_to_action)
