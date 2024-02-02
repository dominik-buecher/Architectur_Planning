import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchinfo import summary
import random
from collections import namedtuple
import numpy as np
import datetime


print("GPU support: ", torch.cuda.is_available())

n_rows = 10
n_cols = 10
N_COWS = 2

from window_richard import *
class GUI():
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Grid Window with Cows, Mower, and Target")
        self.winHandle = GridWindow(self.root, n_rows, n_cols, N_COWS)

    def update(self):
        # self.winHandle.move_cows()
        pass

    def run(self):
        self.root.mainloop()

    def set_mower_abs(self, row, col):
        self.winHandle.move_mower_abs(row, col)

    def get_cow_positions(self):
        return self.winHandle.get_cow_positions()
    
    
    def reset(self):
        self.winHandle.move_mower_abs(0,0)
        self.winHandle.reset()
    
    def move_cows(self):
        self.winHandle.move_cows()


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

# class DQN(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(DQN, self).__init__()
#         # self.fc1 = nn.Linear(input_size, 128)
#         # self.fc2 = nn.Linear(128, 128)
#         # self.fc3 = nn.Linear(128, output_size)

#         # works quite well
#         self.fc1 = nn.Linear(input_size, 128)
#         self.fc2 = nn.Linear(128, output_size)
#         self.fc1.reset_parameters()
#         self.fc2.reset_parameters()

#     def forward(self, x):
#         # x = F.relu(self.fc1(x.view(-1, self.fc1.in_features)))
#         # hidden = self.fc2(x)
#         # return self.fc3(x)

#         x = F.relu(self.fc1(x.view(-1, self.fc1.in_features)))
#         # x = F.sigmoid(self.fc1(x.view(-1, self.fc1.in_features)))
#         # x = self.fc1(x.view(-1, self.fc1.in_features))
#         return self.fc2(x)

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        # self.fc1 = nn.Linear(input_size, 128)
        # self.fc2 = nn.Linear(128, 128)
        # self.fc3 = nn.Linear(128, output_size)

        # works quite well
        self.fc1 = nn.Linear(input_size*2, 128)
        self.fc2 = nn.Linear(128, output_size)
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, x):
        # x is an array of mower-position coords and cow-positions coords
        
        global N_COWS
        
        # calculate distance to COWs
        for i in range(N_COWS):
            # X distance
            x[i+0] = x[i+0] - x[0]
            # Y distance
            x[i+1] = x[i+1] - x[1]
        # distances can be negative, therefore encode with sigmoid rather than relu
        x = F.sigmoid(self.fc1(x.view(-1, self.fc1.in_features)))

        # x = F.sigmoid(self.fc1(x.view(-1, self.fc1.in_features)))
        # x = self.fc1(x.view(-1, self.fc1.in_features))
        return self.fc2(x)

class DQNOneHot(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNOneHot, self).__init__()
        # self.fc1 = nn.Linear(input_size, 128)
        # self.fc2 = nn.Linear(128, 128)
        # self.fc3 = nn.Linear(128, output_size)

        # works quite well
        self.fc1 = nn.Linear(input_size*2*10, 128) # 10 times, as input will be one-hot-encoded
        self.fc2 = nn.Linear(128, output_size)
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, x):
        # x is an array of mower-position coords and cow-positions coords
        
        global N_COWS
        
        # get an input
        input_tensor = x.view(-1, 2 + 2*N_COWS)
        input_tensor = input_tensor.long()

        # encode coords one-hot
        one_hot_encoded = F.one_hot(input_tensor, num_classes=10).float()
        # print(one_hot_encoded)
        
        x = F.sigmoid(self.fc1(one_hot_encoded.view(-1, self.fc1.in_features)))

        # x = F.sigmoid(self.fc1(x.view(-1, self.fc1.in_features)))
        # x = self.fc1(x.view(-1, self.fc1.in_features))
        return self.fc2(x)





class PathFinder():
    def __init__(self):
        # Define the environment
        self.n_rows = 10
        self.n_cols = 10
        # n_states = n_rows * n_cols
        self.n_actions = 5
        self.actions = ['Up', 'Down', 'Left', 'Right', 'Stop']
        self.ACTION_UP    = 0
        self.ACTION_DOWN  = 1
        self.ACTION_LEFT  = 2
        self.ACTION_RIGHT = 3
        self.ACTION_STOP  = 4
        
        self.goal_field = (self.n_rows-1, self.n_cols-1)
        self.goal_reward = self.n_rows * self.n_cols * 2 - 2
        
        self.goal_reached_counter = 0

        self.fields_travelled = {}

    # fields_travelled = []
        
    def get_ideal_action(self, row_idx, col_idx):
        ideal_action = 0
        if row_idx in [0,2,4,6,8]:
            # move right is preferred
            if col_idx == 9:
                # move down is preferred
                ideal_action = self.ACTION_DOWN
            else:
                ideal_action = self.ACTION_RIGHT
        elif row_idx in [1,3,5,7]:
            # move left is preferred
            if col_idx == 0:
                # move down is prefered
                ideal_action = self.ACTION_DOWN
            else:
                ideal_action = self.ACTION_LEFT
        elif row_idx in [9]:
            # move right is preferred
            ideal_action = self.ACTION_RIGHT
        return ideal_action

    def is_allowed_action(self, row, col, action):
        if action == self.ACTION_DOWN:
            next_row = row+1
            next_col = col
        elif action == self.ACTION_UP:
            next_row = row-1
            next_col = col
        elif action == self.ACTION_LEFT:
            next_row = row
            next_col = col-1
        elif action == self.ACTION_RIGHT:
            next_row = row
            next_col = col+1
        elif action == self.ACTION_STOP:
            next_row = row
            next_col = col
        if next_row >= 0 and next_row < self.n_rows and next_col >= 0 and next_col < self.n_cols:
            return True
        else:
            return False
        
    def calculate_next_state_by_row_col(self, row, col, action):
        # Define the transition rules based on the chosen action
        # row, col = current_state // self.n_cols, current_state % self.n_cols

        if action == self.ACTION_UP:  # Move Up
            row = row - 1
            row = max(0, row)
        elif action == self.ACTION_DOWN:  # Move Down
            row = row + 1
            row = min(row, self.n_rows-1)
        elif action == self.ACTION_LEFT:  # Move Left
            col = col - 1
            col = max(0,col)
        elif action == self.ACTION_RIGHT:  # Move Right
            col = col + 1
            col = min(col, self.n_cols-1)
        elif action == self.ACTION_STOP:  # Stop
            pass

        return row, col

    def would_hit_cow(self, row_idx, col_idx, action):
        next_row, next_col = self.calculate_next_state_by_row_col(row_idx, col_idx, action)
        if (next_row, next_col) in g.get_cow_positions():
            return True
        else:
            return False
        
    def has_hit_any_field(self):
        temp = (len(self.fields_travelled.keys()) == 100)
        if temp:
            print("GOAL GOAL GOAL")
        return temp


    def get_reward_for_field(self, row_idx, col_idx, action):
        reward = 0
        ideal_action = self.get_ideal_action(row_idx, col_idx)
        
        if self.is_allowed_action(row_idx, col_idx, action):
            if self.is_goal_reached(row_idx, col_idx):
                # are_all_fields_travelled = (len(self.fields_travelled.keys()) == 100)
                # if are_all_fields_travelled:
                #     reward = 300
                if not (row_idx, col_idx) in self.fields_travelled:
                    if( len(self.fields_travelled.keys()) > 60 ):
                        reward = len(self.fields_travelled.keys()) * 3
                        self.fields_travelled[(row_idx, col_idx)] = True

            else:
                if self.would_hit_cow(row_idx, col_idx, action):
                    reward += -10
                    print(datetime.datetime.now().strftime("%M:%S"), "Cow hit.....")
                if (row_idx, col_idx) in self.fields_travelled:
                    reward += -5
                else:
                    if action == ideal_action:
                        reward += 8
                    elif action == 4: #stop
                        reward += -4
                    else:
                        reward += -11
        else:
            reward = -40

        return reward


        # if rewards[row_idx][col_idx] == 0:
        #     return -1
        # else:
        # r = self.rewards[row_idx][col_idx]
        # self.rewards[row_idx][col_idx] = -1
        # return r

    # def is_goal_reached(self, reward):
    #     if reward == self.goal_reward:
    #         self.goal_reached_counter += 1
    #         # print("goal reached!!")
    #         return True
    #     else:
    #         return False
    
    def is_goal_reached(self, row, col):
        goal_row, goal_col = self.goal_field
        if goal_row == row and goal_col == col:
            self.goal_reached_counter += 1
            # print("goal reached!!")
            return True
        else:
            return False
        

    def run(self):
        # Define the action-to-index dictionary
        index_to_action = {self.ACTION_UP: 'Up', self.ACTION_DOWN: 'Down', self.ACTION_LEFT:'Left', self.ACTION_RIGHT: 'Right', self.ACTION_STOP: 'Stop'}

        # Define rewards and penalties
        



        # DQN parameters
        # gamma = 0.99995    # Discount factor
        gamma = 0.7    # Discount factor
        epsilon = 1.0      # Initial exploration rate
        epsilon_decay = 0.999998
        min_epsilon = 0.03
        # guided_epsilon_probability = 0.9
        guided_epsilon_probability = 0.3
        # epsilon_decay = 0.999998
        # min_epsilon = 0.1
        alpha = 0.20
        batch_size = 32
        target_update = 100
        max_episodes = 8500
        learning_rate = 0.0003
        # max_episodes = 4000

        # Initialize replay memory
        replay_memory_size = 4000
        replay_memory = []

        # Define named tuple for experience replay
        Transition = namedtuple('Transition', ('state_idx', 'state_indexes_cows', 'action', 'next_state_idx', 'reward', 'done'))

        # Initialize DQN and target DQN
        input_size = 1+N_COWS  # State_idx_mower + 4x State_idx_cow
        output_size = self.n_actions
        # dqn = DQN(input_size, output_size)
        # target_dqn = DQN(input_size, output_size)
        dqn = DQNOneHot(input_size, output_size)
        target_dqn = DQNOneHot(input_size, output_size)
        target_dqn.load_state_dict(dqn.state_dict())
        target_dqn.eval()
        summary(target_dqn)

        # Define optimizer and loss function
        optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()

        def calculate_next_state(current_state, action):
            # Define the transition rules based on the chosen action
            # row, col = current_state // self.n_cols, current_state % self.n_cols
            row, col = get_coord_from_state(current_state)

            if action == self.ACTION_UP:  # Move Up
                row = row - 1
                row = max(0, row)
            elif action == self.ACTION_DOWN:  # Move Down
                row = row + 1
                row = min(row, self.n_rows-1)
            elif action == self.ACTION_LEFT:  # Move Left
                col = col - 1
                col = max(0,col)
            elif action == self.ACTION_RIGHT:  # Move Right
                col = col + 1
                col = min(col, self.n_cols-1)
            elif action == self.ACTION_STOP:  # Stop
                pass

            return row, col

        def get_state_from_coord(row, col):
            return row * self.n_cols + col
        
        def get_coord_from_state(state):
            row = int(state/self.n_cols)
            col = int(state % n_rows)
            return row, col

        # Function to select epsilon-greedy action
        def epsilon_greedy_action(state, epsilon, guided_epsilon_probability=0.0):
            if random.uniform(0, 1) < epsilon:
                # Take random action
                if random.uniform(0, 1) < guided_epsilon_probability:
                    r,c = get_coord_from_state(state)
                    action = self.get_ideal_action(r,c)
                else:
                    while True:
                        action = random.randint(0, self.n_actions - 1)
                        # next_row, next_col = calculate_next_state(state, action)
                        # if is_allowed_action(next_row, next_col):
                        #     break
                        r,c = get_coord_from_state(state)
                        if self.is_allowed_action(r,c,action):
                            break

            else:
                # Take action proposed by DQN
                # print("Taking proposed action..")
                state_tensor = torch.tensor(
                    list(get_coord_from_state(state)) + 
                    [xy for coords in g.get_cow_positions() for xy in coords], dtype=torch.float32)  # Wrap the state in a tensor
                # print(state_tensor)
                q_values = dqn(state_tensor)
                max_action = q_values.argmax().item()
                # print("q_values.shape = ", q_values.shape)
                # print("q_values", q_values)
                # print(max_action)
                assert(max_action >= 0 and max_action < self.n_actions)

                next_row, next_col = calculate_next_state(state, max_action)

                # If action is not allowed, anyways take allowed random action
                r,c = get_coord_from_state(state)
                if not self.is_allowed_action(r,c, max_action):
                # if not is_allowed_action(next_row, next_col):
                    while True:
                        action = random.randint(0, self.n_actions - 1)
                        next_row, next_col = calculate_next_state(state, action)
                        if self.is_allowed_action(r,c, action):
                        # if is_allowed_action(next_row, next_col):
                            break
                else:
                    action = max_action
            return action


        action_counter = [0,0,0,0,0]



        def perform_action(state_idx, action, sleep_time_ms=0):
            next_row, next_col = calculate_next_state(state_idx, action)
            next_state_idx = get_state_from_coord(next_row, next_col)
            
            global winHandle
            g.set_mower_abs(next_row, next_col)
            if random.uniform(0,1) < 0.30:
                g.move_cows()
            time.sleep(sleep_time_ms/1000)
            # g.update()
            self.fields_travelled[(next_row, next_col)] = True
            return next_state_idx, next_row, next_col

        def reset_env():
            g.reset()
            self.fields_travelled = {} 
            self.fields_travelled[(0,0)] = True
            
            # g.update()
        
        def do_test():
            
            reset_env()

            # After training, generate actions for an episode
            generated_actions = []
            state = 0  # Starting state
            done = False
            step_count = 0

            while not done:
                state_tensor = torch.tensor(
                    list(get_coord_from_state(state)) + 
                    [xy for coords in g.get_cow_positions() for xy in coords], dtype=torch.float32)  # Wrap the state in a tensor
                # print(state_tensor)
                q_values = dqn(state_tensor)
                print(q_values)
                action = q_values.argmax().item()
                next_row, next_col = calculate_next_state(state, action)
                action = torch.topk(q_values, 2).indices[0, 1].item() if next_row < 0 or next_row >= self.n_rows or next_col < 0 or next_col >= self.n_cols else action
                generated_actions.append(index_to_action[action])
                r,c = get_coord_from_state(state)
                # print(r, c, index_to_action[action])
                # print("Next Position:", next_row, next_col)
                next_state_idx, next_row, next_col = perform_action(state, action, sleep_time_ms=100)

                # next_row, next_col = calculate_next_state(state, action)
                # next_state_idx = next_row * self.n_cols + next_col
                
                done = True if (next_row == self.goal_field[0] and next_col == self.goal_field[1]) else False  # Check if goal reached

                state = next_state_idx
                step_count += 1
                done = True if step_count > 110 else self.has_hit_any_field()
                # print("finished")
                # print("step_count", step_count)
                # print("(next_row == self.goal_field[0] and next_col == self.goal_field[1])", (next_row == self.goal_field[0] and next_col == self.goal_field[1]))

            print("Generated Actions:", generated_actions)
        

        # DQN training loop
        for episode_idx in range(max_episodes):
            row = 0
            col = 0
            total_reward = 0
            done = False
            step_count = 0
            reset_env()


            while not done:
                state_idx = get_state_from_coord(row, col)
                # print("state_idx = ", state_idx)
                action = epsilon_greedy_action(state_idx, epsilon, guided_epsilon_probability=guided_epsilon_probability)
                action_counter[action] += 1
                #print(state_idx, index_to_action[action])
                
                # Simulate environment (transition to next state and get reward)

                next_state_idx, next_row, next_col = perform_action(state_idx, action)
                # next_row, next_col = calculate_next_state(state_idx, action)
                # next_state_idx = next_row * self.n_cols + next_col
                
                # reward = rewards[next_row][next_col]
                reward = self.get_reward_for_field(row, col, action)
                # reward = self.get_reward_for_field(next_row, next_col, action)
                done = self.is_goal_reached(row, col)                    # Check if goal reached
                # done = self.is_goal_reached(reward)                    # Check if goal reached
                total_reward += reward
                # print("Got reward", reward, "for moving", index_to_action[action])
                
                # Store transition in replay memory
                transition = Transition(state_idx, [get_state_from_coord(c[0], c[1]) for c in g.get_cow_positions()], action, next_state_idx, reward, done)
                # print("Adding transition", transition)
                # print(g.get_cow_positions())
                replay_memory.append(transition)
                if len(replay_memory) > replay_memory_size:
                    replay_memory.pop(0)  # Remove oldest transition if memory is full

                # Sample a random batch from replay memory
                if len(replay_memory) > batch_size:
                    transitions = random.sample(replay_memory, batch_size)
                    # state_batch = torch.tensor([t.state_idx for t in transitions], dtype=torch.float32)
                    # for t in transitions:
                    #     print(t.state_indexes_cows)
                    state_batch = torch.tensor([[t.state_idx] + (t.state_indexes_cows) for t in transitions], dtype=torch.float32)
                    state_batch = torch.tensor(
                        [
                            list(get_coord_from_state(t.state_idx)) + 
                            [xy for state_idxes in t.state_indexes_cows for xy in get_coord_from_state(state_idxes)]
                            for t in transitions
                        ], dtype=torch.float32
                    )

                    # list(get_coord_from_state(state)) + 
                    # [xy for coords in g.get_cow_positions() for xy in coords], dtype=torch.float32)  # Wrap the state in a tensor


                    action_batch = torch.tensor([t.action for t in transitions], dtype=torch.int64).unsqueeze(1)
                    reward_batch = torch.tensor([t.reward for t in transitions], dtype=torch.float32)
                    next_state_batch = torch.tensor(
                        [
                            list(get_coord_from_state(t.next_state_idx)) +
                            [xy for state_idxes in t.state_indexes_cows for xy in get_coord_from_state(state_idxes)]
                            for t in transitions
                        ], 
                        dtype=torch.float32
                    )
                    done_batch = torch.tensor([t.done for t in transitions], dtype=torch.float32)
                    # print("state_batch = ", state_batch.shape)
                    # print("action_batch = ", action_batch.shape)
                    # print("reward_batch = ", reward_batch.shape)
                    # print("next_state_batch = ", next_state_batch.shape)

                    # Calculate Q-values for current and next states
                    q_values = dqn(state_batch)
                    # print("Learning batch..")
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

                if step_count > 400:  # Terminate the episode if steps exceed 20
                    done = True
                else:
                    done = self.has_hit_any_field()

                # Decay epsilon
                epsilon = max(min_epsilon, epsilon * epsilon_decay)

            if episode_idx % target_update == 0:
                target_dqn.load_state_dict(dqn.state_dict())  # Update target network

            if ((episode_idx+1) % 100) == 0:
                print("Number of trained episodes: ", episode_idx+1)
                print("Current epsilon =", epsilon)
                print("Goal reached:", self.goal_reached_counter, "times")
                time.sleep(2.5)
                do_test()
                time.sleep(2.5)
                

        print("                                 ", self.actions)
        print("Actions performed during training", action_counter)

        while True:
            print("Testing after training is finished:\n\n")
            time.sleep(20)
            do_test()

        


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