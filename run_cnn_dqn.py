from my_ccpp_map import envi
from cnn_dqn import cdqn
import numpy as np
import torch
from cnn_dqn import turn_input
from tqdm import tqdm
import matplotlib.pyplot as plt
from path_plot import draw_trajectory as dr
from tridional_ways import triditon_ccpp as bow
# def turn_input(input):
#     input = np.array(input)
#     # print("in", input)
#     input = torch.from_numpy(input.reshape(1, 1, map_size+1, map_size))
#     input = (input).double()
#     input = input.to(torch.float32)
#     input = torch.unsqueeze(input, dim = 0)
#     return input
    
def run(max_episode):
    step = 0
    with tqdm(total = max_episode, colour = 'BLUE') as pbar:
        pbar.set_description('training')
        for episode in range(max_episode):
            state = map.reset()
            # print(episode,state)
            every_episode_step = 0
            rate = episode / max_episode
            while True:
                state = turn_input(state, n_state)
                # print("1",state.shape)
                action = model.choose_action(state, rate)
                # print("1")
                state_ , reward, terminal, modiflier = map.step(action)
                model.store_memory(state.flatten(), action, reward, state_.flatten())
                state = state_
                every_episode_step += 1
                
                if terminal:
                    step += 1
                    if step >= 20 and step % 2 == 0:
                        model.learn()
                    break
            pbar.update(1)

if __name__ == '__main__':
     
    batch_size = 32
    memory_size = 2000
    input_channels = 1
    gamma = 0.8
    max_episode = 10
    map_size = 11
    map = envi(map_size, map_size)
    n_state = map.env_size
    # print(n_state)
    memory_length = n_state
    n_actions = map.action_size
    model = cdqn(input_channels, n_actions, batch_size, memory_size, memory_length, gamma)
    run(max_episode)
    #### draw high moves
    map.plt() 
    figure_1 = dr()
    figure_1.record_map(map.print_map())
    steps = map.print_step()
    figure_1.record_actions(steps)
    plt.figure(1)
    plt.subplot(2, 3, 4)
    figure_1.draw_all()
    ####draw cost
    plt.figure(1)
    plt.subplot(2, 3, 5)
    model.draw_cost()
    ###bow
    bow_1 = bow(map.print_map(), [0,0])
    path = bow_1.bow_ccpp()
    print("path", path)
    plt.figure(2)
    figure_2 = dr()
    figure_2.record_map(map.print_map())
    for i in path:
        figure_2.record_path(i)
    figure_2.draw_all()
    plt.show()
