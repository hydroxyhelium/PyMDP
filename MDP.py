## this is a class used to describe Markov Descision Process (MPD)  which forms the backbone for RNN and 
## Reinforcement Learning
from transition_function import TF
from reward import Reward
from policy import Policy
import random

import numpy as np
import copy

random.seed(0)

class MDP:
    ## upon initilization we form a random policy  
    def __init__(self, S: list, A: list, tf:TF, s0: str, r: Reward, alpha: int) -> None:
        self.S = S 
        self.A = A 
        self.tf = tf
        self.s0 = s0 
        self.r = r

        ## Note that alpha is the discount factor 
        self.alpha = alpha

        ## if user doesn't assign a policy we will create a random policy to start with 

        mapping = {}

        action_list = [x for x in self.A]

        for state in S:
            mapping[state] = action_list[random.randint(0, len(A)-1)]
        
        self.policy:Policy = Policy(mapping)

    def change_mapping(self, mapping) -> None:
        self.policy = Policy(mapping)
        return

    def evaluate_policy_finite(self, p:Policy, horizon: int, state: str):
        state_list = [state for state in self.S]
        ##print(state_list)
        hashmap = {} ## maps states to integer to construct a DP table

        for index, value in enumerate(state_list):
            hashmap[index]= value         

        ## we generate a DP table
        dp_table = [ [0 for i in range(len(state_list))] for j in range(horizon+1)]

        ##print(self.tf.get_transition_values('0','c', '3'))       

        for i in range(1, horizon):
            for j in range(len(state_list)):

                temp = 0 
                for l in range(len(state_list)):

                              
                    ##print(str(i)+" "+str(hashmap[j])+" "+str(hashmap[l])+" "+str(dp_table[i-1][l])+" "+str(self.tf.get_transition_values(hashmap[j], p.get_action(hashmap[j]), hashmap[l])))
                    temp += self.tf.get_transition_values(hashmap[j], p.get_action(hashmap[j]), hashmap[l])*dp_table[i-1][l]

                ##print("temp "+str(temp+self.r.get_reward(hashmap[j], p.get_action(hashmap[j]))))
                dp_table[i][j] = self.r.get_reward(hashmap[j], p.get_action(hashmap[j]))+temp

        
        index_associated = state_list.index(state)

        final_res_temp = 0

        for i in range(len(state_list)):
            ##print(dp_table[horizon-1][i]*self.tf.get_transition_values(state, p.get_action(state), hashmap[i]))
            final_res_temp +=  self.tf.get_transition_values(state, p.get_action(state), hashmap[i])*dp_table[horizon-1][i]
        
        dp_table[horizon][index_associated] = self.r.get_reward(state, p.get_action(state))+final_res_temp  

        
        return dp_table[horizon][index_associated]

    def evaulate_policy_infinite(self, p:Policy, state:str): 
        ## to evaluate a policy over infinite horizon, we need to solve a system of equations 
        
        ## Matrix X would contain all the coefficents of state values to be multiplied to 

        X: np.ndarray = np.zeros([len(self.S), len(self.S)])
        Y: np.ndarray = np.zeros([len(self.S), 1]) 
        res: np.ndarray = np.zeros([len(self.S), 1])

        lst = list(self.S)
        special_index = 0 ## this is the index to which the state we're intrested in gets mapped to

        hashmap = {}

        for index,value in enumerate(lst):
            if(value==state):
                special_index = index
            hashmap[index]=value

        for i in range(len(lst)):
            Y[i][0]=self.r.get_reward(hashmap[i], p.get_action(hashmap[i]))
            for j in range(len(lst)):
                if(j==0):
                    X[i][j]=1
                
                else:
                    X[i][j]= - self.alpha*self.tf.get_transition_values(hashmap[i], p.get_action(hashmap[i], hashmap[j]))
        
        res = np.linalg.inv(X)@Y

        return res[special_index][0]

    def find_optimal_policy_finite(self, state:str, horizon: int)->Policy:

        ## given horizon and starting state finds the optimal policy to acculamate greatest number of expected rewards. 
        ## uses 3-Dim DP table

        optimal_policy = {} ## we start out with an empty dict.

        state_list = list(self.S)
        action_list = list(self.A)

        hashmap_states = {}
        hashmap_actions = {}

        for index, state in enumerate(state_list):
            hashmap_states[index]=state

        for index, action in enumerate(action_list):
            hashmap_actions[index]=action

        dp_table = [[[0 for i in range(state_list)] for j in range(action_list)] for k in range(horizon+1)]

        for height in range(horizon+1):

            for action in range(action_list):
                for state in range(state_list):

                    temp=0 
                    
                    for i in range(state_list):
                        temp_table = [ x[i] for x in dp_table[height-1]]
                        temp += self.tf.get_transition_values(state_list[state],action_list[action] ,state_list[i])*max(temp_table)    
                    dp_table[height][action][state]= self.r.get_reward(action_list[action], state_list[state])+temp

        
        temp = np.array(dp_table[horizon]).argmax(axis=0)

        for i in range(len(state_list)):
            optimal_policy[state_list[i]]=action_list[temp[i]]
    
        return Policy({optimal_policy})

    def find_optimal_policy_infinite(self, epsilon):
        ## optimal_policy in case of infinte horizon would be a stationary. 
        hashmap = {}

        for i in (self.S):
            for j in (self.A):
                hashmap[(i,j)]=0

        old_hashmap = copy.deepcopy(hashmap) 


        all_tuples = []

        for i in self.S:
            for j in self.A:
                all_tuples.append((i,j))

        while True:
            for i in (self.S):
                for j in (self.A):

                    temp = 0 

                    for l in (self.S):
                        temp += self.alpha*self.tf(i, j, l)*max([old_hashmap[(l, x)] for x in self.A])


                    hashmap[(i,j)]=self.r(i, j)+temp
                        

            if(max([abs(hashmap[x]-old_hashmap[x]) for x in all_tuples]) < epsilon):

                policy_mapping = {}

                for state in self.S:
                    array_values = np.array([hashmap[(state, action)] for action in self.A])
                    index_highest = np.argmax(array_values)
                    policy_mapping[state]=self.A[index_highest]

                return Policy(policy_mapping) 

            old_hashmap = copy.deepcopy(hashmap)


