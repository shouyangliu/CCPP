import numpy as np
import matplotlib.pyplot as plt

class c_map():
    def __init__(self, h, l):
        self.h = h
        self.l = l
        self.reward_list = []
        self.pace = []
        self.build_map()

    def build_map(self):
        self.map = np.zeros((self.h, self.l))
        obscale = [(2,2), (2,3)]
        for i in obscale:
            self.map[i] = -1
        self.map[0, 0] = 7
        return self.map

    def reset(self):
        self.step = 0
        self.build_map()
        return self.map

    ##
    # @brief 
    #
    # @param action [0,1,2,3] -> [up, down, right, left]
    #
    # @return 
    def take_action(self, action):
        state = np.pad(self.map, (1,1), 'constant', constant_values = -1)
        robot_position = np.array(np.where(state == 7)).flatten()
        direction = np.array([0, 0])
        if action == 0:
            direction = np.array([0, -1])
        elif action == 1:
            direction = np.array([0, 1])
        elif action == 2:
            direction = np.array([1, 0])
        elif action == 3:
            direction = np.array([-1, 0])
        else:
            print("the action is error and the action is ", action)
        new_position = robot_position + direction
        self.step += 1

        ###
        new_position = (new_position[0], new_position[1])
        robot_position = (robot_position[0], robot_position[1])
        if state[new_position] == -1:
            reward = -100
            done = True
        elif state[new_position] == 0:
            reward = 20
            done = False
        elif state[new_position] == 1:
            reward = 0
            done = False
            for i in [-1, 1]:
                if (state[robot_position[0]+i, robot_position[1]] != 1 or -1) \
                or (state[robot_position[0], robot_position[1]+i] != 1 or -1):
                    reward = -20

        if self.step > 2*self.l * self.h:
            done = True
            reward = 0

        if done:
            self.reward_list.append(np.sum(self.map>=1))
            self.pace.append(self.step)
            state_ = np.zeros((self.h, self.l))
        else:
            robot_position = np.array(robot_position) - np.array([1,1])
            new_position = np.array(new_position) - np.array([1,1])
            robot_position = (robot_position[0], robot_position[1])
            new_position = (new_position[0], new_position[1])
            self.map[robot_position] = 1
            self.map[new_position] = 7
            
            state_ = self.map

        return state_, reward, done

    def draw_cover_rate(self):
        plt.title("cover_rate")
        plt.xlabel("times")
        plt.plot(self.reward_list, label = 'reward', color = 'b')

    def draw_pace(self):
        plt.title("pace")
        plt.xlabel("times")
        plt.plot(self.pace, label = 'pace', color = 'r')











