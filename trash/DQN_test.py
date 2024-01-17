import tensorflow as tf
import numpy as np

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def act(self, state):
        q_values = self.model.predict(np.array([state]))
        return np.argmax(q_values[0])

    def train(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = (reward + 0.95 * np.amax(self.model.predict(np.array([next_state]))[0]))
        target_f = self.model.predict(np.array([state]))
        target_f[0][action] = target
        self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

# Beispiel für die Nutzung des DQN-Agenten
state_size = # Größe deines Zustands
action_size = # Anzahl der Aktionen
agent = DQNAgent(state_size, action_size)
num_episodes = 50
max_timesteps = 


# Trainingsschleife
for episode in range(num_episodes):
    state = # Initialisierung des Zustands
    for time in range(max_timesteps):
        action = agent.act(state)
        next_state, reward, done, _ = # Führe die ausgewählte Aktion in der Umgebung aus
        agent.train(state, action, reward, next_state, done)
        state = next_state
        if done:
            break
