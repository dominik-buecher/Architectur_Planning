import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchinfo import summary
import random
from collections import namedtuple
import numpy as np


print("GPU support: ", torch.cuda.is_available())

n_rows = 10
n_cols = 10

from window import *
class GUI():
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Grid Window with Cows, Mower, and Target")
        self.winHandle = GridWindow(self.root, n_rows, n_cols, 4)

    def update(self):
        # self.winHandle.move_cows()
        pass

    def run(self):
        self.root.mainloop()

    def set_mower_abs(self, row, col):
        self.winHandle.move_mower_abs(row, col)


g = GUI()




# generate file
# import os
# import os.path
# if not os.path.isdir("./outputs"):
#     os.makedirs("./outputs")
# import time
# import sys
# f = open("./outputs/output_"+time.strftime("%Y%m%d-%H%M%S")+".txt", "w+", encoding="utf8")
# orig_stdout = sys.stdout
# sys.stdout = f

# print script content
# with open(os.path.abspath(__file__), "r") as sc:
#     print(sc.read())

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        # self.fc1 = nn.Linear(input_size, 128)
        # self.fc2 = nn.Linear(128, 128)
        # self.fc3 = nn.Linear(128, output_size)

        # works quite well
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        # x = F.relu(self.fc1(x.view(-1, self.fc1.in_features)))
        # hidden = self.fc2(x)
        # return self.fc3(x)

        x = F.relu(self.fc1(x.view(-1, self.fc1.in_features)))
        # x = F.sigmoid(self.fc1(x.view(-1, self.fc1.in_features)))
        # x = self.fc1(x.view(-1, self.fc1.in_features))
        return self.fc2(x)

class PathFinder():
    def __init__(self):
        # Define the environment
        self.n_rows = 10
        self.n_cols = 10
        # n_states = n_rows * n_cols
        self. n_actions = 4 
        self.actions = ['Up', 'Down', 'Left', 'Right']
        
        self.rewards = []
        self.goal_field = (self.n_rows-1, self.n_cols-1)
        self.goal_reward = 100
        
        self.goal_reached_counter = 0

    def reset_rewards(self):
        self.rewards = np.zeros((self.n_rows, self.n_cols))
        self.rewards[0][1]   = 9
        self.rewards[0][2]   = 9
        self.rewards[0][3]   = 9
        self.rewards[0][4]   = 9
        self.rewards[0][5]   = 9
        self.rewards[0][6]   = 9
        self.rewards[0][7]   = 9
        self.rewards[0][8]   = 9
        self.rewards[0][9]   = 9
        self.rewards[1][9]   = 9
        self.rewards[2][9]   = 9
        self.rewards[3][9]   = 9
        self.rewards[4][9]   = 9
        self.rewards[5][9]   = 9
        self.rewards[6][9]   = 9
        self.rewards[7][9]   = 9
        self.rewards[8][9]   = 9
        self.rewards[self.goal_field[0]][self.goal_field[1]] = self.goal_reward

    # fields_travelled = []
    def get_reward_for_field(self, row_idx, col_idx):
        # if rewards[row_idx][col_idx] == 0:
        #     return -1
        # else:
        r = self.rewards[row_idx][col_idx]
        self.rewards[row_idx][col_idx] = -1
        return r

    def is_goal_reached(self, reward):
        if reward == self.goal_reward:
            self.goal_reached_counter += 1
            # print("goal reached!!")
            return True
        else:
            return False

    def run(self):
        # Define the action-to-index dictionary
        index_to_action = {0: 'Up', 1: 'Down', 2:'Left', 3: 'Right'}

        # Define rewards and penalties
        
        self.reset_rewards()



        # DQN parameters
        gamma = 0.99995    # Discount factor
        epsilon = 1.0      # Initial exploration rate
        epsilon_decay = 0.999997
        min_epsilon = 0.1
        alpha = 0.20
        batch_size = 32
        target_update = 100
        max_episodes = 2000
        # max_episodes = 4000

        # Initialize replay memory
        replay_memory_size = 4000
        replay_memory = []

        # Define named tuple for experience replay
        Transition = namedtuple('Transition', ('state_idx', 'action', 'next_state_idx', 'reward', 'done'))

        # Initialize DQN and target DQN
        input_size = 1  # Only the state index
        output_size = self.n_actions
        dqn = DQN(input_size, output_size)
        target_dqn = DQN(input_size, output_size)
        target_dqn.load_state_dict(dqn.state_dict())
        target_dqn.eval()
        summary(target_dqn)

        # Define optimizer and loss function
        optimizer = optim.Adam(dqn.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()

        def calculate_next_state(current_state, action):
            # Define the transition rules based on the chosen action
            row, col = current_state // self.n_cols, current_state % self.n_cols

            if action == 0:  # Move Up
                row = row - 1
                row = max(0, row)
            elif action == 1:  # Move Down
                row = row + 1
                row = min(row, self.n_rows-1)
            elif action == 2:  # Move Left
                col = col - 1
                col = max(0,col)
            elif action == 3:  # Move Right
                col = col + 1
                col = min(col, self.n_cols-1)

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
            if next_row >= 0 and next_row < self.n_rows and next_col >= 0 and next_col < self.n_cols:
                return True
            else:
                return False

        # Function to select epsilon-greedy action
        def epsilon_greedy_action(state, epsilon):
            if random.uniform(0, 1) < epsilon:
                # Take random action
                while True:
                    action = random.randint(0, self.n_actions - 1)
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
                        action = random.randint(0, self.n_actions - 1)
                        next_row, next_col = calculate_next_state(state, action)
                        if is_allowed_action(next_row, next_col):
                            break
                else:
                    action = max_action
            return action


        action_counter = [0,0,0,0]

        def perform_action(state_idx, action):
            next_row, next_col = calculate_next_state(state_idx, action)
            next_state_idx = next_row * self.n_cols + next_col
            
            global winHandle
            g.set_mower_abs(next_row, next_col)
            time.sleep(0.05)
            # g.update()
            return next_state_idx, next_row, next_col

        def reset_env():
            global winHandle
            g.set_mower_abs(0, 0)
            # g.update()

        # DQN training loop
        for episode_idx in range(max_episodes):
            row = 0
            col = 0
            total_reward = 0
            done = False
            step_count = 0
            self.reset_rewards()
            reset_env()


            while not done:
                state_idx = row * self.n_cols + col
                # print("state_idx = ", state_idx)
                action = epsilon_greedy_action(state_idx, epsilon)
                action_counter[action] += 1
                #print(state_idx, index_to_action[action])
                
                # Simulate environment (transition to next state and get reward)

                next_state_idx, next_row, next_col = perform_action(state_idx, action)
                # next_row, next_col = calculate_next_state(state_idx, action)
                # next_state_idx = next_row * self.n_cols + next_col
                
                # reward = rewards[next_row][next_col]
                reward = self.get_reward_for_field(next_row, next_col)
                done = self.is_goal_reached(reward)                    # Check if goal reached
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

                if step_count > 100:  # Terminate the episode if steps exceed 20
                    done = True

                # Decay epsilon
                epsilon = max(min_epsilon, epsilon * epsilon_decay)

            if episode_idx % target_update == 0:
                target_dqn.load_state_dict(dqn.state_dict())  # Update target network

            if ((episode_idx+1) % 1000) == 0:
                print("Number of trained episodes: ", episode_idx+1)
                print("Current epsilon =", epsilon)
                print("Goal reached:", goal_reached_counter, "times")

        print("                                 ", self.actions)
        print("Actions performed during training", action_counter)

        # After training, generate actions for an episode
        generated_actions = []
        state = 0  # Starting state
        done = False
        step_count = 0

        while not done:
            q_values = dqn(torch.tensor([state], dtype=torch.float32))
            print(q_values)
            action = q_values.argmax().item()
            next_row, next_col = calculate_next_state(state, action)
            action = torch.topk(q_values, 2).indices[0, 1].item() if next_row < 0 or next_row >= self.n_rows or next_col < 0 or next_col >= self.n_cols else action
            generated_actions.append(index_to_action[action])
            print(int(state/self.n_rows), state%self.n_cols, index_to_action[action])
            print("Next Position:", next_row, next_col)

            next_row, next_col = calculate_next_state(state, action)
            next_state_idx = next_row * self.n_cols + next_col
            
            done = True if (next_row == goal_field[0] and next_col == goal_field[1]) else False  # Check if goal reached

            state = next_state_idx
            step_count += 1
            done = True if step_count > 50 else done

        print("Generated Actions:", generated_actions)


import threading
import time

p = PathFinder()

def thready():
    p.run()
    # for i in range(1,100):
        # time.sleep(1)
        # g.update()
        # p.

t = threading.Thread(target=thready)
t.start()

g.run()








# sys.stdout = orig_stdout
# f.close()