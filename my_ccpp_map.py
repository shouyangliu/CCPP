import numpy as np
import matplotlib.pyplot as plt

rewa = []
episode_steps = []
total_score = []

goal = "ccpp"
#goal = "target"

class envi():
    '''class envi:the environment
    env_size 11*10
    action_size 3
    '''
    def __init__(self, h, l) -> None:
        self.h = h
        self.l = l
        if goal == "ccpp":
            self.env_size = self.h
            print("self.ebv", self.env_size)
        if goal == "target":
            self.env_size = 2
        self.action_size = 4
        self.highest_reward = 0
        self.shortest_steps = 2 * self.h*self.l
        self.highest_moves = []
        self.build_map()
        self.highest_score = 0
        self.pre_highest_map = self.map
        
        
    def build_map(self):
        self.map = np.zeros((self.h + 1, self.l))
        ##### build obstacle
        #self.map[2,3] = -1
        for i in range(6,10):
            self.map[2, i] = -1
            self.map[3, i] = -1
        for i in range(3):
            self.map[3, i] = -1
        for i in range(4):
            self.map[4, i] = -1

        self.map[7, 5] = -1
        for i in range(8,10):
            self.map[6, i] = -1
            self.map[7, i] = -1
        ##### end of obstacle
        
        self.static_map = self.map[0:-1, :]
        self.map[0, 0] = 1   #start point
        self.map[-1,0:2] = [0, 0]   ## axis
        self.step_c = 0
        self.obstacle = np.sum(self.map[:-1, :] == -1)
        self.goal_point = [4,4]# [self.h - 2, self.l -2]
        self.full_score = np.sum(self.map[:-1, :] != -1)
        self.moves = []
        self.position = [0, 0]
        
        
    def print_step(self):
        return self.highest_moves   
        
    def print_map(self):
        return self.static_map
        

    def reset(self):
        '''[summary] reset the maze 
        Returns:
            [self.map]: 11*10 matrix
        '''
        self.build_map()
        
        #return self.map
        if goal == "ccpp":
            return self.map 
        if goal == "target":
            return (self.goal_point[0] - 1, self.goal_point[1] - 1)
    
    def step(self, action):
        '''[summary] 

        Args:
            action (one of [0,1,2,3]): [0 is up;1 is down;2 is left; 3 is right]

        Returns:
            [s_, reward, done, self.position]: [description]
        '''
        #s = self.map
        #self.position = self.map[self.h , 0:2].astype(int)
        self.step_c += 1
        self.extra_reward = 0
        self.moves.append(action)
        
        cross = False
        if(action == 0 ):#and self.position[0] > 1):  #up
            if self.position[0] > 1:
                self.position[0] -= 1
            else:
                cross = True

        if(action == 1 ):#and self.position[0] < self.h -2):  #down
            if self.position[0] < (self.h -1):
                self.position[0] += 1  
            else:
                cross = True
                  
        if(action == 2 ):#and self.position[1] > 2):  #left
            if self.position[1] > 1:
                self.position[1] -= 1
            else:
                cross = True
                
        if(action == 3 ):#and self.position[1] < self.l - 2):  #right
            if self.position[1] < (self.l -1):
                self.position[1] += 1
            else:
               cross = True
           
        self.extra_reward = 10 * np.sum(self.map >=1) - 7 * self.step_c
        
        if goal == "ccpp":
            #hit the obstacle
            if(self.map[self.position[0], self.position[1]] == -1 or cross):
                reward = -1
                done = True
            
            else:
                done = False
                reward = 0
                self.map[self.h,0] = self.position[0]
                self.map[self.h,1] = self.position[1]

                self.map[self.position[0],self.position[1]] = 1 #self.position[0] * self.h + self.position[1] 

                state_ = self.map
                
                # ## go into the blank one
                #if(self.map[self.position[0], self.position[1]] == 0):
                #    reward = 1 
                #    done = False
                #else:## go into the black one
                #    #no other way to go 
                #    if(self.map[abs(self.position[0]-1), self.position[1]] != 0 and
                #       self.map[self.position[0], abs(self.position[1]-1)] != 0 and 
                #       self.map[min(self.position[0]+1, self.h -1), self.position[1]] != 0 and
                #       self.map[self.position[0], min(self.position[1]+1, self.l -1)] != 0 ):
                #        reward = 0
                #        done = False
                    #there is blank one it can go into
                #    else:
                #        reward = -20
                #        done = False
                 
                # # cover the hole map    
                if(np.sum(self.map[:-1,:] == 0) == 0):
                    reward = 30#10 * np.sum(self.map[:-1,:] >=1) - 5 * self.step_c + 100 
                    done = True
                elif (self.map == self.highest_map).all() and (self.map != self.static_map).all():
                    reward = 10
                    done = False
                # # #over the max steps 
                elif(np.sum(self.step_c) >= 5 * (self.h * self.l - self.obstacle)):
                    reward = -1
                    done = True
                
                else:
                    reward = -1
                    done = False


            if done:  
                rewa.append(reward)
                episode_steps.append(self.step_c)
                self.score = 10 * np.sum(self.map[:-1, :] >= 1) - self.step_c
                total_score.append(self.score)
                if self.score >= self.highest_score:
                    self.highest_score = self.score
                    self.pre_highest_map = self.map
                    self.highest_moves = self.moves
                    
                state_ = np.zeros((self.h+1, self.l))
                #rewa.append(np.sum(self.map[:-1,:] >=1) - 0.1 * self.step_c)
                #print("clc")
                self.step_c = 0
                self.moves = []
                
            return state_, reward, done, 0#self.final_reward
                
                
    def change_hightest_map(self):
        self.highest_map = self.pre_highest_map
    
    def plt_reward(self):
        plt.title("reward")
        plt.xlabel("times")
        plt.ylabel("grade")
        # plt.axhline(self.full_score, color = 'r', linestyle = '-')
        x = []
        for i in range(len(rewa)):
            x.append(i)
        
        # plt.plot(x, episode_steps, label = "step", color = 'r')
        plt.plot(x, rewa, label = "grade")
    
    def plt_grade(self):
        # print("highest reward", self.highest_reward) 
        plt.title("grade")
        plt.xlabel("times")
        plt.ylabel("grade")
        plt.plot(total_score, label = "grade")
