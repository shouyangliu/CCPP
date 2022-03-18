import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from random import choice
import random
import matplotlib.pyplot as plt

class c_Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.conv2d_layer = nn.Sequential(\
                nn.Conv2d(in_channels = input_dim, out_channels = 6, kernel_size = 3, stride = 1, padding = 2),\
                nn.ReLU(), \
                nn.Conv2d(6, 3, 3),
                nn.ReLU(),)

        self.fc = nn.Sequential(\
                nn.Linear(300, 150),\
                nn.ReLU(),\
                nn.Linear(150,40),\
                nn.ReLU(),\
                nn.Linear(40, output_dim))

    def forward(self, DQN_input):
        x = torch.from_numpy(DQN_input)
        x = x.to(torch.float32)
        x = self.conv2d_layer(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x

class c_improved_dqn():
    def __init__(self, input_dim, output_dim, batch_size, memory_size, gamma):
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.gamma = gamma
        self.memory_counter = 0
        self.learn_counter = 0

        self.eval_net = c_Net(input_dim, output_dim)
        self.target_net = c_Net(input_dim, output_dim)
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr = 0.001)

        self.memory_state = np.zeros((self.memory_size, 10, 10))
        self.memory_state_ = np.zeros((self.memory_size, 10, 10))
        self.memory_others = np.zeros((self.memory_size, 2))

    def choose_action(self, state, rate):
        if random.random() < rate:
            state = state.reshape((1,1,10,10))  # (batch_size, height, width, depth)
            action_value = self.eval_net(state)
            action = torch.argmax(action_value)
        else:
            p_state = np.pad(state, (1,1), 'constant', constant_values = (-1, -1))
            robot_position = np.array(np.where(p_state == 7))
            local_map = p_state[int(robot_position[0]-1):int(robot_position[0]+2), int(robot_position[1]-1):int(robot_position[1]+2)]
            action_space = []
            loss = [(0,1), (2,1), (1,0), (1,2)]
            action_list = [0, 1, 2, 3]
            for i in range(4):
                if local_map[loss[i]] != -1:
                    action_space.append(action_list[i])
            action = choice(action_space)
        
        return action

    def store_momery(self, state, state_, action, reward):
        index = int(self.memory_counter % self.memory_size)
        self.memory_state[index,:,:] = state
        self.memory_state_[index,:,:] = state_
        self.memory_others[index, :] = np.array([reward, action])
        self.memory_counter += 1

    def learn(self):
        if self.learn_counter % 200 == 0:
            self.target_net.load_state_dict((self.eval_net.state_dict()),False)
        self.learn_counter += 1

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size = self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size = self.batch_size)
        state = self.memory_state[sample_index,:].reshape(self.batch_size, 1, 10, 10)
        state_ = self.memory_state_[sample_index, :].reshape(self.batch_size, 1, 10, 10)
        reward = torch.LongTensor(self.memory_others[sample_index, 0])
        action = torch.LongTensor(self.memory_others[sample_index, 1].reshape(32, 1))
        q_eval = self.eval_net(state).gather(1, action)
        q_target = self.target_net(state_).detach()
        q_target_max = torch.max(q_target, dim = 1)[0]
        q = reward + self.gamma * (q_target_max)
        loss = self.loss(q_eval, q.reshape(self.batch_size, 1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
