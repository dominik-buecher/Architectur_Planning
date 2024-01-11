import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[random.randrange(self.capacity)] = transition

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

class Agent:
    def __init__(self, input_size, output_size, gamma=0.99, epsilon=0.1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(input_size, output_size).to(self.device)
        self.target_net = DQN(input_size, output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = ReplayMemory(10000)
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randrange(self.policy_net.fc3.out_features)  # Exploration
        else:
            with torch.no_grad():
                q_values = self.policy_net(torch.tensor(state, dtype=torch.float32).to(self.device))
                return torch.argmax(q_values).item()  # Exploitation

    def optimize_model(self):
        if len(self.memory.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)
        batch = list(zip(*transitions))

        state_batch = torch.tensor(batch[0], dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(batch[1], dtype=torch.long).to(self.device)
        reward_batch = torch.tensor(batch[2], dtype=torch.float32).to(self.device)
        next_state_batch = torch.tensor(batch[3], dtype=torch.float32).to(self.device)

        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()

        expected_q_values = reward_batch + self.gamma * next_q_values

        loss = F.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Beispiel für die Verwendung:
env = YourCustomEnvironment(rows=20, cols=20, num_cows=5)  # Du musst deine eigene Umgebung erstellen
input_size = env.observation_space.shape[0]  # Anzahl der Merkmale im Zustand
output_size = env.action_space.n  # Anzahl der Aktionen

agent = Agent(input_size, output_size)

for episode in range(NUM_EPISODES):
    state = env.reset()
    total_reward = 0

    while True:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)

        agent.memory.push((state, action, reward, next_state))
        agent.optimize_model()

        total_reward += reward
        state = next_state

        if done:
            break

    if episode % TARGET_UPDATE == 0:
        agent.target_net.load_state_dict(agent.policy_net.state_dict())

    print(f"Episode {episode}, Total Reward: {total_reward}")
