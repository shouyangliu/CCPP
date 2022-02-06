import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt

cost = [] #store each trained data's cost

#####只负责神经网络部分
def turn_input(input, map_size):
    input = np.array(input)
    # print("in", input)
    input = torch.from_numpy(input.reshape(1, 1, map_size+1, map_size))
    input = (input).double()
    input = input.to(torch.float32)
    input = input
    # input = torch.unsqueeze(input, dim = 0)
    return input

class Net(nn.Module):
    def __init__(self, input_channels, output_dim) -> None:
        '''[summary]

        Args:
            input_channels ([int]): [RGB pitcure is 3]
            output_dim ([int]): [action dim]
        '''
        super().__init__()
        
        self.Conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels = input_channels, out_channels=6, kernel_size = 5, stride=1, padding=2),
            nn.ReLU(),
            #nn.MaxPool2d(2, 2)
        ) 
        
        self.Conv_layer_2 = nn.Sequential(
            nn.Conv2d(6, 3, 3),
            nn.ReLU(),
            #nn.MaxPool2d(2, 2)
        )       
        
        self.fully_connected_1 = nn.Sequential(
            nn.Linear(270, 126),
            # nn.BatchNorm1d(12),
            nn.ReLU()
        )
        
        self.fully_connected_2 = nn.Sequential(
            # nn.BatchNorm1d(4),
            nn.Linear(126, 18),
            nn.ReLU(),
            nn.Linear(18, output_dim)
        )
        
        
    def forward(self, x):
        x = torch.tensor(x)
        # print("x",x.shape)
        x = self.Conv_layer_1(x)
        x = self.Conv_layer_2(x)
        x = x.view(x.size()[0], -1)
        x = self.fully_connected_1(x)
        x = self.fully_connected_2(x)
        # print("xxx", x) 
        return x
    
class cdqn():
    def __init__(self, input_channels, output_dim,
                 batch_size, memory_size, memory_length, gamma) -> None:
        '''[define the net, create the memory bank]

        Args:
            input_channels ([int]): [the input channels]
            output_dim ([int]): [action dim]
            batch_size ([int]): [batch]
            memory_size ([int]): [the length of the memory bank]
            memory_length ([int]): [the length of each memory]
        '''
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval_net = Net(input_channels, output_dim)
        self.target_net = Net(input_channels, output_dim)
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr = 0.5)
        
        self.gamma = gamma        
        self.output_dim = output_dim
        self.memory_counter = 0
        self.learn_step = 0
        self.map_size = memory_length
        self.memory_length = memory_length * (memory_length + 1)
        self.memory = np.zeros((self.memory_size, self.memory_length * 2 +2))
    
    def choose_action(self, x, rate):
        if random.random() < rate:
            action_value = self.eval_net(x)
            # print("work", action_value)
            action =  torch.max(action_value, 1)[1].data.numpy()[0]
        
        else:
            action = np.random.randint(0, self.output_dim)
        
        return action

    def store_memory(self, state, action, reward, state_ ):
        transition = np.hstack((state, [ action , reward ], state_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1
        
    def learn(self):
        if self.learn_step % 200 == 0:
            self.target_net.load_state_dict((self.eval_net.state_dict()), False)
        self.learn_step += 1
        
        ## 抽取记忆
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        memory = self.memory[sample_index, :]  # 抽取的记忆单元，并逐个提取
        state = memory[:, :self.memory_length]
        action = memory[:, self.memory_length : self.memory_length+1]
        reward = memory[:, self.memory_length+1 : self.memory_length+2]
        state_ = memory[:, -self.memory_length:]
        # print("memory_length", self.memory_length)
        action = torch.LongTensor(action)
        reward = torch.LongTensor(reward)
        # state = turn_input(state, self.map_size)
        # state_ = turn_input(state_, self.map_size)
        x = torch.empty(self.batch_size, self.map_size+1, self.map_size)
        x_ = torch.empty(self.batch_size, self.map_size+1, self.map_size)
        # print(x.shape, state.shape)
        for i in range(self.batch_size):
            x[i,:] = torch.tensor(turn_input(state[i,:], self.map_size))
            x_[i, :] = torch.tensor(turn_input(state_[i,:], self.map_size))
        x = x.unsqueeze(dim = 1)
        x_ = x_.unsqueeze(dim = 1)
        # print("wwww",x.shape)
        q_eval = self.eval_net(x).gather(1, action)
        # print("Sds")
        q_target = self.target_net(x_).detach()
        q = reward + self.gamma * (q_target.max(1)[0].unsqueeze(1)) 
        
        loss = self.loss(q_eval, q)
        cost.append(loss)
        # 反向传播更新
        self.optimizer.zero_grad()  # 梯度重置
        loss.backward()  # 反向求导
        self.optimizer.step()  # 更新模型参数
        
    def draw_cost(self):
        plt.xlabel('steps')
        plt.ylabel('cost')
        plt.title('cost')
        plt.plot(cost) 
