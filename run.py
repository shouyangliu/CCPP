import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from environment import c_map
from improved_dqn import c_improved_dqn

def run(max_episode):
    episode_step = 0
    with tqdm(total = max_episode, colour = 'BLUE') as pbar:
        pbar.set_description('training')
        for episode in range(max_episode):
            state = grid_map.reset()
            rate = episode / max_episode
            while True:
                action = model.choose_action(state, rate)
                state_, reward, terminal = grid_map.take_action(action)
                model.store_momery(state, state_, action, reward)
                state = state_
                if episode_step > 200 and episode_step % 5 == 0:
                    model.learn()
                if terminal:
                    break
                episode_step += 1

if __name__ == '__main__':
    map_size = 10
    batch_size = 32
    memory_size = 2000
    input_dim = 1
    output_dim = 4
    gamma = 0.8
    max_episode = 200000
    grid_map = c_map(map_size, map_size)
    model = c_improved_dqn(input_dim, output_dim,batch_size, memory_size, gamma)
    run(max_episode)

    plt.figure(1)
    plt.subplot(2,3,1)
    grid_map.draw_cover_rate()

    plt.subplot(2,3,2)
    grid_map.draw_pace()
    plt.show()
