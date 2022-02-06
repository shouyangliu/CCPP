import numpy as np
import queue
from my_ccpp_map import envi
from path_plot import draw_trajectory
import matplotlib.pyplot as plt

class A_star():
    def __init__(self, map):
        """!
        @brief [need input a matrix map]
        Paramètres : 
            @param map => [a matrix :-1->obstacle, 1->passed grid, 0->blank grid]
        """
        self.map = map
            
    def distance(self, pos):
        """!
        @brief [calculate the distance of point to goal]
        Paramètres : 
            @param pos => [two elements list]

        """
        return (self.goal[0] - pos[0] + self.goal[1] - pos[1])

    def find_nlist(self, point):
        """!
        @brief [find the passable grid around the point (just 4 directions)]

        Paramètres : 
            @param point => [two elements list]

        """
        nlist = []
        ## down search
        if point[0] > 0 and self.map[point[0] - 1, point[1]] != -1:
            nlist.append([point[0]-1, point[1]])
        ## laft search
        if point[1] > 0 and self.map[point[0], point[1] - 1] != -1:
            nlist.append([point[0], point[1]-1])
        ### up search
        if point[0] < (self.map.shape[0] - 1) and self.map[point[0]+1, point[1]] != -1:
            nlist.append([point[0]+1, point[1]])
        ### right search
        if (point[1] < self.map.shape[1] -1) and self.map[point[0], point[1]+1] != -1:
            nlist.append([point[0], point[1]+1])
        
        return nlist


    def a_star_search(self, start, goal):
        """!
        @brief [search the shortest path between the two point]

        Paramètres : 
            @param start:[two elements list]
            @param goal:[two elements list]
        """
        self.start = start 
        self.goal = goal
        frontier = queue.PriorityQueue()
        frontier.queue.clear()
        frontier.put(self.start, 0)
        came_from = {}
        cost_so_far = {}
        came_from[tuple(self.start)] = None  #record father node
        cost_so_far[tuple(self.start)] = 0
        path = []

        while not frontier.empty():
            current = frontier.get()
            if current == self.goal:
                step = self.goal
                path.append(step)
                while step != self.start:
                    step = came_from[tuple(step)]
                    path.append(step)
                break
            for next in self.find_nlist(current):
                new_cost = cost_so_far[tuple(current)] + 1#graph.cost(current, next)
                if tuple(next) not in cost_so_far or new_cost < cost_so_far[tuple(next)]:
                    cost_so_far[tuple(next)] = new_cost
                    priority = new_cost + self.distance(next)
                    frontier.put(next, priority)
                    came_from[tuple(next)] = current
        return path



class triditon_ccpp():
    def __init__(self, environment, position):
        """!
        @brief [the bow ccpp]
        example:a = triditon_ccpp(map,start_point)
                b = a.bow_ccpp()
                #the b is the path
        Paramètres : 
            @param environment => [the map]
            @param position => [the position of start point]

        """
        self.map = environment
        self.position = position

    def max_index(self, index):
        """!
        @brief [Description de la fonction]

        Paramètres : 
            @param self => [description]
            @param index => [description]

        """
        '''
        only works when map's h equal to map's l
        '''
        return min(index, self.map.shape[0] - 1)
    def min_index(self, index):
        """!
        @brief [Description de la fonction]

        Paramètres : 
            @param self => [description]
            @param index => [description]

        """
        return max(0, index)

    def bow_ccpp(self):
        point = []
        find_way = A_star(self.map)
        point.append(self.position)
        self.map[self.position[0], self.position[1]] = 1
        while True:
            if np.sum(self.map == 0) == 0:
                break
            #### if the up and right grid is empty,then get into it
            if self.position[1] < self.map.shape[1] - 1 and \
                    self.map[self.position[0], self.position[1] + 1] != -1 and \
                    self.map[self.position[0], self.position[1] + 1] != 1: #if up can step into a new grid
                self.position = [self.position[0], self.position[1]+1]  # up
                point.append(self.position)
            elif self.position[1] > 0 and \
                    self.map[self.position[0], self.position[1] - 1] != -1 and \
                    self.map[self.position[0], self.position[1] - 1] != 1: #if down can step into a new grid
                self.position = [self.position[0], self.position[1] - 1]
                point.append(self.position)
            elif self.position[0] < self.map.shape[0] - 1 and \
                    self.map[self.position[0] + 1, self.position[1]] != -1 and \
                    self.map[self.position[0] + 1, self.position[1]] != 1: #if right can step into a new grid
                self.position = [self.position[0] + 1, self.position[1]] 
                point.append(self.position)
            
            else:#search the closed new grid and use A* to reach it
                blank_point = []
                for i in range(self.map.shape[0]):
                    x_pos_add = self.max_index(self.position[0] + i)
                    x_pos_min = self.min_index(self.position[0] - i)
                    y_pos_add = self.max_index(self.position[1] + i)
                    y_pos_min = self.min_index(self.position[1] - i)

                    #firstly search the horizontal and vertical
                    if self.map[x_pos_add, self.position[1]] == 0:#right
                        blank_point = [x_pos_add, self.position[1]] 
                    if self.map[x_pos_min, self.position[1]] == 0:#left
                        blank_point = [x_pos_min, self.position[1]] 
                    if self.map[self.position[0], y_pos_min] == 0:#down
                        blank_point = [self.position[0], y_pos_min]
                    if self.map[self.position[0], y_pos_add] == 0:#up
                        blank_point = [self.position[0], y_pos_add]

                    ##diagonal
                    if self.map[x_pos_add, y_pos_add] == 0:
                        blank_point = [x_pos_add, y_pos_add] 
                    if self.map[x_pos_add, y_pos_min] == 0:
                        blank_point = [x_pos_add, y_pos_min] 
                    if self.map[x_pos_min, y_pos_add] == 0:
                        blank_point = [x_pos_min, y_pos_add] 
                    if self.map[x_pos_min, y_pos_min] == 0:
                        blank_point = [x_pos_min, y_pos_min] 
                    for j in range(i):
                        if self.map[self.max_index(self.position[0]+j), y_pos_min] == 0:
                            blank_point = [self.max_index(self.position[0]+j), y_pos_min]
                        if self.map[self.min_index(self.position[0]-j), y_pos_min] == 0:
                            blank_point = [self.min_index(self.position[0]-j), y_pos_min]
                        if self.map[x_pos_add, self.max_index(self.position[1]+j)] == 0:
                            blank_point = [x_pos_add, self.max_index(self.position[1]+j)]
                        if self.map[x_pos_min, self.min_index(self.position[1]-j)] == 0:
                            blank_point = [x_pos_min, self.min_index(self.position[1]-j)]
                        if self.map[self.max_index(self.position[0]+j), y_pos_add] == 0:
                            blank_point = [self.max_index(self.position[0]+j), y_pos_max]
                        if self.map[self.min_index(self.position[0]-j), y_pos_add] == 0:
                            blank_point = [self.min_index(self.position[0]-j), y_pos_add]
                        if self.map[x_pos_min, self.max_index(self.position[1]+j)] == 0:
                            blank_point = [x_pos_min, self.max_index(self.position[1]+j)]
                        if self.map[x_pos_add, self.min_index(self.position[1]-j)] == 0:
                            blank_point = [x_pos_add, self.min_index(self.position[1]-j)] 
                    if blank_point != []:
                        break
                path = find_way.a_star_search(self.position, blank_point)
                #path  = find_way.a_star_search([0,0], [2,3])
                for i in range(2, len(path) + 1):
                    point.append(path[-i])
                self.position = blank_point
            self.map[self.position[0], self.position[1]] = 1
        return point



#a = np.zeros((6,6))
#a[1,1] = -1
#a[5,5] = -1
#c = triditon_ccpp(a, [0,0])
#path = c.bow_ccpp()
#plt.figure(1)
#d = draw_trajectory()
#d.record_map(a)
#for i in path:
#    d.record_path(i)
#plt.show()
