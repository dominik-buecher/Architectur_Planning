import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import namedtuple
import tkinter as tk
from window import *


rows = 20
cols = 20
n_states = rows * cols
n_actions = 4 
num_cows = 2
input_size = (n_states + 4 + (2 * num_cows))  # Zustand + Positionen der Kühe
output_size = n_actions


root = tk.Tk()
root.title("Grid Window with Cows, Mower, and Target")
# Erstellen des Enviroments
grid_window = GridWindow(root, rows, cols, num_cows)


# DQN model
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Erfahrung wird als Transition abgepeichert -> wird für Memory gebraucht
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

# ReplayMemory wird verwendet damit der Agent aus alten Erfahrungen lernen kann
# Erklärung -> https://deeplizard.com/learn/video/Bcuj2fTH4_4
# -> https://www.kaggle.com/code/viznrvn/deep-q-learning-with-experience-replay-theory
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    # push fügt eine neue Erfahrung/Transition zur Memory hinzu
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    # Wählt zufällig eine Stichprobe von Erfahrungen aus der Memory
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

# Definiere den DQN agent
# openai gym agent oder stable baseline agent ausprobieren
class DQNAgent:
    def __init__(self, input_size, output_size, capacity=10000, batch_size=32, gamma=0.999, epsilon=1.0, epsilon_decay=0.999, min_epsilon=0.1):
        # Parameter festlegen
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.output_size = output_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size

        # Erstelle DQN um Aktionen basierend auf dem aktuellen Zustand vorherzusagen
        self.policy_net = DQN(input_size, output_size).to(self.device)
        
        # Erstelle DQN um Zielwerte für das Q-Lernverfahren bereitzustellen
        # -> Gewichte des Target-Netzwerks werden periodisch mit den Gewichten des Policy-Netzwerks synchronisiert
        self.target_net = DQN(input_size, output_size).to(self.device)
        
        # Target Netz mit Gewichten des Policy Netzes synchronisieren
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Wird in Evaluierungsmodus gesetz damit die Gewichte nicht aktualisiert werden
        self.target_net.eval()
        
        # Optimierer für das Policy Netz 
        # -> Optimierer optimiert die Gewichte des Netzes aufgrund der Loss Funktion
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        
        # Memory erstellen mit Maixmalen Wert an Erfahrungen die gesammelt werden können
        self.memory = ReplayMemory(capacity)
    
    
    # Zufällige Aktion auswählen oder Aktion wählen die den Höchsten q-Value hat
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.output_size - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor([state], dtype=torch.float32).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
    
    
    # trainiert den agent -> Zufälliger Wert aus Memory wählen zum verbessern
    def train(self):
        if len(self.memory.memory) < self.batch_size:
            return
        
        # Wählt zufällige Erfahrung aus Memory 
        transitions = self.memory.sample(self.batch_size)

        # Erstellt Liste/Batch mit den Transtions
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
        loss.backward()
        self.optimizer.step()
    # Aktualisieren des Netzes mit den neuen Gewichten
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    # Reduzieren des Wahrscheinlichkeit für eine zufällige aktion
    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


n_actions = 4
index_to_action = {0: 'Up', 1: 'Down', 2:'Left', 3: 'Right'}


# Create the DQN agent
dqn_agent = DQNAgent(input_size, output_size)

train = True
#train = False

# training ausführen
if train is True:
    initial_state = grid_window.get_state()
    max_episodes = 1000
    for episode in range(max_episodes):
        # aktuellen state laden
        state = initial_state
        grid_window.reset_to_initial_state(initial_state)
        total_reward = 0
        done = False
        print("episode: ", episode)
        action_counter = 0
        # Schleife durchlaufen bis bedingung erfüllt ist
        while not done:
            # Wähle Aktion basierend auf aktuellem Zustand
            action = dqn_agent.select_action(state)
            action_counter += 1
            # Erhalte zukünfitgen State welcher durch die Aktion erreicht wird
            next_row, next_col = grid_window.get_future_state(state, action)

            grid_window.move_mower_abs(next_row, next_col)
            grid_window.move_cows()
            grid_window.root.update()
            grid_window.root.after(1)

            next_state = grid_window.get_state()  # Update the state
            # Berechne Reward basierend auf dem zukünftigen Zustand 
            reward = grid_window.get_reward(state, next_row, next_col)
            
            # Wenn Reward = 500 erreicht ist oder 1000 Aktionen durchgeführt wurden -> Schleife abbrechen
            done = True if reward >= 500 or action_counter >= 10000 else False
            #print("reward: ", reward)
            
            # Fügt den durchlauf als Transition in die Memory hinzu
            dqn_agent.memory.push(state, action, next_state, reward, done)
            
            # Train Funktion von Agent aufrufen
            dqn_agent.train()
            total_reward += reward
         
            #print("action: ", action)
            state = next_state

        # Verringern der Wahrscheinlichkeit für zufällige Aktionen
        dqn_agent.decay_epsilon()
        
        # Nach 10 Episoden wird das Netz geupdated
        if episode % 10 == 0:
            dqn_agent.update_target_net()

    # trainiertes Modell speichern
    torch.save(dqn_agent.policy_net.state_dict(), 'models/dqn_model.pth')

else:
    # trainierted Modell laden
    loaded_state_dict = torch.load('models/dqn_model.pth')
    dqn_agent.policy_net.load_state_dict(loaded_state_dict)

# Wird verwendet um die SChritte des Agent in dem Enviorment darzustellen
def play_environment(grid_window, actions, index_to_action):
    for action in actions:
        print("action: ", action)
        
        if action in index_to_action.values():
            # prüfen ob Aktion im Dictionary enthalten ist
            action_index = next((index for index, act in index_to_action.items() if act == action), None)
            
            # Wähle neuen Zustand basierend auf der Aktion
            future_row, future_col = grid_window.get_future_state(grid_window.get_state(), action_index)
            
            # Bewege Rasenmäher zu neuem Zustand
            grid_window.move_mower_abs(future_row, future_col)
        
        grid_window.move_cows()
        grid_window.root.update()
        grid_window.root.after(500)

        # Überprüfe, ob alle Felder besucht und Ziel erreicht
        if grid_window.get_state()[-1] == 1 and grid_window.get_state()[-2] == 1:
            print("Spiel beendet! Ziel erreicht und alle Felder besucht.")
            break




generated_actions = []
state = grid_window.get_state()
done = False
total_reward = 0

while not done:
    # Aktion wählen
    action = dqn_agent.select_action(state)
    
    # Neuen Zustand wählen basierend auf der Aktion
    next_row, next_col = grid_window.get_future_state(state, action)
    next_state = grid_window.get_state()
    
    # Alle Aktionen abspeichern
    generated_actions.append(index_to_action[action])
    state = next_state
    
    # Reward berechnen basierend auf der Aktino
    reward = grid_window.get_reward(state, next_row, next_col)
    total_reward += reward
    
    # Abbruchbedingung prüfen: Ziel erreicht? oder Reward über 100
    done = True if state[-1] == 1 or total_reward > 100 else False  # Check if target reached or reward > 1000
    print("total_reward: ", total_reward)
    
print("Generated Actions:", generated_actions)

play_environment(grid_window, generated_actions, index_to_action)
