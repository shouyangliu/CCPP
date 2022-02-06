from random import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
#np.set_printoptions(threshold=np.inf)
from my_ccpp_map import envi
from path_plot import draw_trajectory
from tqdm import tqdm

max_episode = 15000
size = 12
learn_reward = []

class Net(nn.Module):
    def __init__(self, in_dim, n_actions) -> None:
        super().__init__()
        in_dim = [(size-2), (size-2)]
            #input: size * size * 1 output: size - 2
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.R1 = nn.ReLU()
        
        self.p1 = nn.MaxPool2d(2)
        self.c2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=1, stride=1, padding=0)
        self. R2 = nn.ReLU()
        
        self.p2 = nn.MaxPool2d(2)
        self.c3 = nn.Conv2d(16, 120, 3, 1, 0)
    
        self.R3 = nn.ReLU()
        self.Linear_layers = nn.Sequential(
           nn.Linear(in_features=120, out_features=n_actions), 
           print("4")
        )
        # ### fully connected
        # layer_size = [n_states, 20, 15, 10, 8, 4, n_actions]
        # for i in range(len(layer_size) - 2):
        #     self.layers.add_module('layer_'+str(i), nn.Linear(layer_size[i], layer_size[i+1]))
        #     self.layers.add_module('ReLu'+str(i), nn.ReLU())
        # self.layers.add_module('layer_'+str(len(layer_size)), nn.Linear(layer_size[-2], layer_size[-1]))
        # ###end of the fully connected
        
        # self.initialize()
        
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
        
    def forward(self, input):
        print("input", input)
        # input = torch.from_numpy(input.reshape(1, 1, size+1, size))
        # input = Variable(input).double()
        # input = input.to(torch.float32)
        # print("input", input.shape)
        out = self.c1(input)
        out = self.R1(out)
        out = self.p1(out)
        out = self.c2(out)
        out = self.R2(out)
        out = self.p2(out)
        out = self.c3(out)
        out = self.R3(out)            
        print("SSS",out)
        out = out.view(out.size(0), -1)
        print("sss", out.size())
        out = self.Linear_layers(out)
        print("out", out)
        
        return out
    
class DQN():
    def __init__(self, n_states, n_actions) -> None:
        self.eval_net, self.target_net = Net(n_states, n_actions), Net(n_states, n_actions)
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr = 0.5)
        #self.optimizer = torch.optim.RMSprop(self.eval_net.parameters(), lr = 0.01)   ## mo fan use it
        self.n_states = n_states
        self.n_actions = n_actions
        self.batch_size = 32
        self.transition_size = 2000
        self.gamma = 0.9
        
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((self.transition_size, self.n_states * 2 +2))
        #print("v",n_states*2+2)
        self.cost = []
        
    def choose_action(self, x, epsilon):
    
        #x = torch.unsqueeze(torch.FloatTensor(x),0)
        x = np.array(x)
        x = torch.from_numpy(x.reshape(1, 1, size+1, size))
        x = Variable(x).double()
        x = x.to(torch.float32)
        #x = torch.FloatTensor(x)
        #print("x",epsilon)
        if True:#random.random() < epsilon: # should be epsilon
            action_value = self.eval_net(x)
            print("action value",action_value)
            action = torch.max(action_value, 1)[1].data.numpy()[0]
            #print("action",action)
        else:
            #print("random choice")
            action = np.random.randint(0, n_actions)
            
        return action
    
    def store_transition(self, state, action, reward, state_ ):
        transition = np.hstack((state.flatten() , [ action , reward ], state_.flatten()))
        # print("tr",transition)
        index = self.memory_counter % self.transition_size
        self.memory[index, :] = transition
        
        self.memory_counter += 1
        #self.memory_counter = self.memory_counter % self.transition_size 
    
    def modify(self, modifier, step_every_episode):
        if step_every_episode >= 2:
            for i in range(step_every_episode - 1):
                self.memory[(self.memory_counter % self.transition_size) - i - 2, n_states + 1] += modifier
                #print("self",self.memory[(self.memory_counter % self.transition_size) - i - 1, n_states + 1 ], modifier)
    
             
        
                
    def learn(self):
        #print("learn")
        if self.learn_step_counter % 200 == 0:
            self.target_net.load_state_dict((self.eval_net.state_dict()), False)
        self.learn_step_counter += 1
        
        # 使用记忆库中批量数据
        #sample_index = np.random.choice(self.transition_size, self.batch_size)  # 2000个中随机抽取32个作为batch_size
        
        if self.memory_counter > self.transition_size:
            sample_index = np.random.choice(self.transition_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        memory = self.memory[sample_index, :]  # 抽取的记忆单元，并逐个提取
        #print("memory",memory)
        state = torch.FloatTensor(memory[:, :n_states])
        #state = (state - 0)/()
        action = torch.LongTensor(memory[:, n_states : n_states+1])
        reward = torch.LongTensor(memory[:, n_states+1:n_states+2])
        next_state = torch.FloatTensor(memory[:, -n_states:])
        print("q_va", self.eval_net(state))
        # 计算loss,q_eval:所采取动作的预测value,q_target:所采取动作的实际value
        q_eval = self.eval_net(state).gather(1, action) # eval_net->(64,4)->按照action索引提取出q_value
        q_dd = self.eval_net(next_state)
        index = torch.max(q_dd, 1)[1].unsqueeze(0)
        q_next = self.target_net(next_state).detach()
        #print("shape",index.size(),q_next.size())
        for i in range(self.batch_size):
            #print("sss",next_state[i, :],"zeros",torch.zeros([next_state.shape[1]]))
            if torch.equal(next_state[i, :], torch.FloatTensor(torch.zeros([next_state.shape[1]]))):
                # print("iiiii",state[i], action[i])
                # print("reward", reward[i])
                q_next[i, :] = torch.zeros([q_next.shape[1]])
                #print("max",q_next[i, :])
            learn_reward.append(memory[i, n_states + 1])   
        # torch.max->[values=[],indices=[]] max(1)[0]->values=[]
        #print("max", torch.gather(q_next, 1, index))
        q_target = reward + self.gamma * (q_next.max(1)[0].unsqueeze(1)) # label
        
        ###DDQN
        #q_target = reward + self.gamma * q_next.gather(1, index)
        #print("max",q_next.max(1)[0].unsqueeze(1))
        #q_target = reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        #print("q",q_eval,"q_target",q_next)
        
        # q_eval = self.eval_net(state)
        # q_next = self.target_net(state).detach()
        # q_target = q_eval
        # batch_index = np.arrange(self.batch_size, dtype=np.int32)
        
        # q_target[batch_index, action] = reward + self.gamma * q_next.max(1)[0].unsqueeze(1)
        # print("q",q_eval,"q_target",q_target,"s",q_next.max(1)[0].unsqueeze(1))
        
        loss = self.loss(q_eval, q_target)
        #print("loss",loss)
        self.cost.append(loss)
        # 反向传播更新
        self.optimizer.zero_grad()  # 梯度重置
        loss.backward()  # 反向求导
        self.optimizer.step()  # 更新模型参数
        
    def print_info(self):
        print("self.counter",self.memory_counter)

    def plot_cost(self):
        plt.figure(1)
        plt.subplot(2, 3, 2)
        plt.plot(np.arange(len(self.cost)), self.cost)
        plt.xlabel("step")
        plt.ylabel("cost")
        #plt.show()    
        
        
def run():
    step = 0
    with tqdm(total = max_episode, colour = 'BLUE') as pbar:
        pbar.set_description('training')
        for episode in range(max_episode):
            state = map.reset()
            step_every_episode = 0
            epsilon = episode / max_episode
            while True:
                action = model.choose_action(state, epsilon)
                #print("action", action)
                state_ , reward, terminal, modiflier = map.step(action)
                model.store_transition(state, action, reward, state_)
                # print("S",(state[:n_states], action, reward, state_[:n_states]))
                # if step > 200 and step % 5 == 0:
                #     model.learn()
                # 进入下一步
                state = state_
                # if terminal:
                #     print("episode=", episode, end=",")
                #     print("step=", step_every_episode,"/",step)
                #     break
                #print("step+1")
                #step += 1
                step_every_episode += 1
                step += 1
                if step > 200 and step % 5 == 0:
                    model.learn() 
                if terminal:
                    # step += 1
                    #print("episode=", episode, end = ",")
                    #print("step=", step_every_episode,"/",step)
                    #model.modify(modiflier, step_every_episode)
                    # if step > 20 and step % 6 == 0:
                    #     model.learn()
                        
                    break
            pbar.update(1)
            
    ##generate the path
    state = map.reset()
    print("draw")
    last_actions = []
    while True:
        action = model.choose_action(state[:n_states], 1)
        last_actions.append(action)
        #0 is up;1 is down;2 is left; 3 is right
        a = ['up', 'down', 'left', 'right']
        print( a[action])
        state_ , reward, terminal, modiflier = map.step(action)
        state = state_
        # if state:
        #     print("opp, hit the obstacle")
        if terminal:
            plt.figure(1)
            plt.subplot(2, 3, 3)
            figure_1.record_actions(last_actions)
            figure_1.draw_all()
            break  
      
    steps = map.print_step()
    figure_1.record_actions(steps)
    plt.figure(1)
    plt.subplot(2, 3, 4)
    figure_1.draw_all()
    plt.figure(1)
    plt.subplot(2, 3, 5)
    plt.plot(learn_reward)
    
            
if __name__ == "__main__":
    
    device = torch.device('cpu')
    
    h = size
    l = size
    map = envi(h,l)
    n_states = map.env_size
    n_actions = map.action_size
    model = DQN(n_states, n_actions)
    figure_1 = draw_trajectory()
    figure_1.record_map(map.print_map()) 
    run()
    model.print_info()
    model.plot_cost()
    map.plt()   
    plt.show()  