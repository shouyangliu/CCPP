import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
from my_ccpp_map import envi



class Net(nn.Module):
    def __init__(self, n_states, n_actions) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_states, 20)
        self.fc2 = nn.Linear(20,30)
        self.fc3 = nn.Linear(30, n_actions)
        self.fc1.weight.data.normal_(0., 0.3)
        self.fc2.weight.data.normal_(0., 0.3)
        self.fc1.bias.data.fill_(0.1)
        self.fc2.bias.data.fill_(0.1)
        self.fc3.weight.data.normal_(0., 0.3)
        self.fc3.bias.data.fill_(0.1)
        
    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        
        return out
    
class DQN():
    def __init__(self, n_states, n_actions) -> None:
        self.eval_net, self.target_net = Net(n_states, n_actions), Net(n_states, n_actions)
        self.loss = nn.MSELoss()
        #self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr = 0.01)
        self.optimizer = torch.optim.RMSprop(self.eval_net.parameters(), lr = 0.01)
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
    
        x = torch.unsqueeze(torch.FloatTensor(x),0)
        #x = torch.FloatTensor(x)
        #print("x",epsilon)
        if np.random.uniform() < epsilon:
            action_value = self.eval_net(x)
            #print("action value",action_value)
            action = torch.max(action_value, 1)[1].data.numpy()[0]
            
            #print("action",action)
        else:
            #print("random choice")
            action = np.random.randint(0, n_actions)
            
        return action
    
    def store_transition(self, state, action, reward, state_ ):
        transition = np.hstack((state , [ action , reward ], state_))
        index = self.memory_counter % self.transition_size
        self.memory[index, :] = transition
        
        self.memory_counter += 1
        #self.memory_counter = self.memory_counter % self.transition_size 
    
    def modify(self, modifier, step_every_episode):
        if step_every_episode >= 2:
            for i in range(step_every_episode):
                self.memory[(self.memory_counter % self.transition_size) - i - 2, n_states + 1] += modifier
                #print("self",self.memory[(self.memory_counter % self.transition_size) - i - 1], modifier)
    
             
        
                
    def learn(self):
        #print("learn")
        if self.learn_step_counter % 200 == 0:
            self.target_net.load_state_dict((self.eval_net.state_dict()))
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
        action = torch.LongTensor(memory[:, n_states : n_states+1])
        reward = torch.LongTensor(memory[:, n_states+1:n_states+2])
        next_state = torch.FloatTensor(memory[:, -n_states:])

        # 计算loss,q_eval:所采取动作的预测value,q_target:所采取动作的实际value
        q_eval = self.eval_net(state).gather(1, action) # eval_net->(64,4)->按照action索引提取出q_value
        q_next = self.target_net(next_state).detach()
        #print("shape",next_state.shape[0])
        for i in range(next_state.shape[0]):
            #print("sss",next_state[i, :],"zeros",torch.zeros([next_state.shape[1]]))
            if torch.equal(next_state[i, :], torch.zeros([next_state.shape[1]])):
                #print("iiiii")
                q_next[i, :] = torch.zeros([q_next.shape[1]])
                #print("max",q_next[i, :])
        # torch.max->[values=[],indices=[]] max(1)[0]->values=[]
        q_target = reward + self.gamma * abs(q_next.max(1)[0].unsqueeze(1)) # label
        
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
        plt.plot(np.arange(len(self.cost)), self.cost)
        plt.xlabel("step")
        plt.ylabel("cost")
        plt.show()    
        
        
def run():
    step = 0 
    max_episode = 100000
    for episode in range(max_episode):
        state = map.reset()
        step_every_episode = 0
        epsilon = episode / max_episode
        while True:
            
            if(n_states == 2):
                action = model.choose_action(state,epsilon)
                state_ , reward, terminal, modiflier = map.step(action)
                model.store_transition(state, action, reward, state_)
            else:
                action = model.choose_action(state.flatten()[:n_states], epsilon)
                state_ , reward, terminal, modiflier = map.step(action)
                model.store_transition(state.flatten()[:n_states], action, reward, state_.flatten()[:n_states])
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
            if terminal:
                step += 1
                if step > 20 :
                    model.learn()
                print("episode=", episode, end=",")
                print("step=", step_every_episode,"/",step)
                model.modify(modiflier, step_every_episode)
                #model.learn()
                break
            
            
if __name__ == "__main__":
    map = envi()
    n_states = map.env_size
    n_actions = map.action_size
    model = DQN(n_states, n_actions)
    run()
    model.print_info()
    model.plot_cost()
    map.plt()     