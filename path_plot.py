import numpy as np
import matplotlib.pyplot as plt

class draw_trajectory():
    def __init__(self) -> None:
        '''the input is the map
        Args:
            map (np.array): cannot be a list, must be an array 
        '''
        self.path = []
    
    def record_map(self, map):
        self.map = map
        # print("map", map)
        
    def draw_obstacle(self):
        ax = plt.gca()
        #print("soze",self.map)
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                if self.map[i][j] == -1:
                    ax.add_patch(plt.Rectangle((i, j), 1, 1, color = "black", fill = True, linewidth = 1))
    
    def record_path(self, position):
        '''record the each step of the trajectory

        Args:
            position (np.array)
        '''
        position = position + np.array([0.5, 0.5])   #the base is the left-up of the square
        self.path.append(position)
        
    def record_actions(self, actions):
        print(actions, len(actions))
        position = np.array([0.5, 0.5])
        self.path.append(position)
        for action in actions:
            if action == 0:#up
                position = position + np.array([-1, 0])
            elif action == 1:
                position = position + np.array([+1, 0])
            elif action == 2:
                position = position + np.array([0, -1])  
            elif action == 3:
                position = position + np.array([0, +1]) 
            self.path.append(position)
        
    def draw_path(self):
        for i in range(len(self.path) - 1):
            dx = self.path[i+1][0] - self.path[i][0]
            dy = self.path[i+1][1] - self.path[i][1]
            #print("ss",self.path[i],self.path[i+1])
            # point_1 = [self.path[i][0], self.path[i+1][0]]
            # point_2 = [self.path[i][1], self.path[i+1][1]]
            # plt.plot(point_1, point_2, color = 'r', markersize = 20)
            plt.arrow(self.path[i][0], self.path[i][1], dx, dy, length_includes_head = True, head_width = 0.1, head_length = 0.2)
        
        
    def prepare_draw_each_move(self):
        self.draw_obstacle()
        my_x_ticks = np.arange(0, self.map.shape[0], 1)
        my_y_ticks = np.arange(0, self.map.shape[1], 1)
        
        plt.grid()
        plt.xlim((0, self.map.shape[0]))
        plt.ylim((0, self.map.shape[1]))
        plt.xticks(my_x_ticks)
        plt.yticks(my_y_ticks)
        plt.gca().set_aspect("equal")
        plt.gca().axes.xaxis.set_ticklabels([])
        plt.gca().axes.yaxis.set_ticklabels([])
        

    def draw_each_move(self, position):
        ax = plt.gca()
        ax.add_patch(plt.Rectangle((position[0], position[1]), 1, 1, color = "red", fill = True, linewidth = 1))


    def end_draw_each_move(self):
        plt.ioff()

    def draw_all(self):
        '''use it you need `plt.figure` firstly
        '''
        #plt.figure()
        self.draw_path()
        self.draw_obstacle()
        my_x_ticks = np.arange(0, self.map.shape[0], 1)
        my_y_ticks = np.arange(0, self.map.shape[1], 1)
        
        plt.grid()
        plt.xlim((0, self.map.shape[0]))
        plt.ylim((0, self.map.shape[1]))
        plt.xticks(my_x_ticks)
        plt.yticks(my_y_ticks)
        plt.gca().set_aspect("equal")
        plt.gca().axes.xaxis.set_ticklabels([])
        plt.gca().axes.yaxis.set_ticklabels([])
        #plt.show()
        self.path = []
