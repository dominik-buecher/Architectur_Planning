import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchinfo import summary
import random
from collections import namedtuple
import numpy as np
import tensorflow as tf
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers



print("GPU support: ", torch.cuda.is_available())

# # generate logfile
# import os
# import os.path
# if not os.path.isdir("./outputs"):
#     os.makedirs("./outputs")
# import time
# import sys
# f = open("./outputs/output_"+time.strftime("%Y%m%d-%H%M%S")+".txt", "w+", encoding="utf8")
# orig_stdout = sys.stdout
# sys.stdout = f
# # print script content
# with open(os.path.abspath(__file__), "r") as sc:
#     print(sc.read())

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

class DQN():
    def create():
        inputs = layers.Input(shape=1)
        layer1 = layers.Dense(128, activation="relu", input_shape=(1,))(inputs)
        layer2 = layers.Dense(64, activation="relu")(layer1)
        action = layers.Dense(n_actions, activation="linear")(layer2)
        return tensorflow.keras.Model(inputs=inputs, outputs=action)


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
gamma = 0.9995          # Discount factor
epsilon = 1.0      # Initial exploration rate
epsilon_decay = 0.9997
min_epsilon = 0.1
alpha = 0.20
batch_size = 32
target_update = 100
max_episodes = 4000
# max_episodes = 4000

# Initialize replay memory
replay_memory_size = 4000
replay_memory = []

# Define named tuple for experience replay
Transition = namedtuple('Transition', ('state_idx', 'action', 'next_state_idx', 'reward', 'done'))

# Initialize DQN and target DQN
input_size = 1  # Only the state index
output_size = n_actions
dqn = DQN.create()
target_dqn = DQN.create()
dqn.summary()


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
    else:
        assert(0)

    return row, col 

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
            action = np.random.choice(n_actions)
            next_row, next_col = calculate_next_state(state, action)
            if is_allowed_action(next_row, next_col):
                break
    else:
        # Take action proposed by DQN
        state_tensor = tf.convert_to_tensor([state])  # Wrap the state in a tensor
        # torch.tensor([state], dtype=torch.float32)  # Wrap the state in a tensor
        q_values = dqn(state_tensor, training = False)
        max_action = tf.argmax(q_values).numpy()
        # max_action = q_values.argmax().item()
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

from tensorflow import keras

optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
loss_function = keras.losses.Huber()

# DQN training loop
for episode_idx in range(max_episodes):
    row = 0
    col = 0
    total_reward = 0
    done = False
    step_count = 0

    while not done:
        state_idx = row * n_cols + col
        # print("state_idx = ", state_idx)
        action = epsilon_greedy_action(state_idx, epsilon)
        action_counter[action] += 1
        #print(state_idx, index_to_action[action])
        
        # Simulate environment (transition to next state and get reward)
        next_row, next_col = calculate_next_state(state_idx, action)
        next_state_idx = next_row * n_cols + next_col
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
            state_batch      = np.array([t.state_idx for t in transitions])
            action_batch     = np.array([t.action for t in transitions])
            reward_batch     =          [t.reward for t in transitions]
            next_state_batch =          [t.next_state_idx for t in transitions]
            done_batch       = np.array([t.done for t in transitions])
            
            # print("state_batch = ", state_batch.shape)
            # print("action_batch = ", action_batch.shape)
            # print("reward_batch = ", reward_batch.shape)
            # print("next_state_batch = ", next_state_batch.shape)

            # Calculate Q-values for current and next states
            future_reward_batch = target_dqn.predict(next_state_batch)
            updated_q_values = reward_batch + gamma * tf.reduce_max(future_reward_batch)
            updated_q_values = updated_q_values * (1 - done_batch) - done_batch
            masks = tf.one_hot(action_batch, n_actions)

            with tf.GradientTape() as tape:
                print(state_batch)
                # q_values = dqn.fit_generator()
                temp = tf.expand_dims(state_batch, 0)
                temp = tf.convert_to_tensor(temp)
                print(state_batch)
                q_values = dqn(temp)
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                loss = loss_function(updated_q_values, q_action)

            grads = tape.gradient(loss, dqn.trainable_variables)
            optimizer.apply_gradients(zip(grads, dqn.trainable_variables))
        
        row = next_row
        col = next_col
        step_count += 1

        if step_count > 40:  # Terminate the episode if steps exceed 40
            done = True

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if episode_idx % target_update == 0:
        target_dqn.set_weights(dqn.get_weights())  # Update target network

    if ((episode_idx+1) % 1000) == 0:
        print("Number of trained episodes: ", episode_idx)
        print("Current epsilon =", epsilon)
        print("Goal reached:", goal_reached_counter, "times")

print("                                 ", actions)
print("Actions performed during training", action_counter)

# After training, generate actions for an episode
generated_actions = []
state = 0  # Starting state
done = False
step_count = 0

while not done:
    state_tensor = tf.convert_to_tensor([state])
    q_values = dqn(state_tensor, training=False)
    action = tf.argmax(q_values).numpy()
    next_row, next_col = calculate_next_state(state, action)

    generated_actions.append(index_to_action[action])
    print(int(state/n_rows), state%n_cols, index_to_action[action])
    print("Next Position:", next_row, next_col)
    next_state_idx = next_row * n_cols + next_col

    done = True if (next_row == goal_field[0] and next_col == goal_field[1]) else False  # Check if goal reached
    state = next_state_idx
    step_count += 1
    done = True if step_count > 50 else done


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

# sys.stdout = orig_stdout
# f.close()