## this is a class used to describe Markov Descision Process (MPD)  which forms the backbone for RNN and 
## Reinforcement Learning
from transition_function import TF
from reward import Reward
from policy import Policy
import random

import numpy as np

random.seed(0)


class MDP:
    ## upon initilization we form a random policy  
    def __init__(self, S: set, A: set, tf:TF, s0: str, r: Reward, alpha: int) -> None:
        self.S = S 
        self.A = A 
        self.tf = tf
        self.s0 = s0 
        self.r = r

        ## Note that alpha is the discount factor 
        self.alpha = alpha

        ## if user doesn't assign a policy we will create a random policy to start with 

        mapping = {}

        action_list = list(A)

        for state in S:
            mapping[state] = action_list[random.randint(0, len(A)-1)]
        
        self.policy:Policy = Policy(mapping)

    def change_mapping(self, mapping) -> None:
        self.policy = Policy(mapping)
        return

    def evaluate_policy_finite(self, p:Policy, horizon: int, state: str) -> int:
        state_list = list(self.S)

        hashmap = {} ## maps states to integer to construct a DP table

        for index, value in enumerate(state_list):
            hashmap[index]= value         

        ## we generate a DP table
        dp_table = [ [0 for i in range(len(state_list))] for j in range(horizon+1)]

        for i in range(1, horizon):
            for j in range(len(state_list)):

                temp = 0 
                for l in range(len(state_list)):
                    temp += self.tf.get_transition_values(hashmap[j], p.get_action(hashmap[j]), hashmap[l])*dp_table[i-1][l]

                dp_table[i][j] = self.r.get_reward(hashmap[j], p.get_action(hashmap[j]))+temp

        
        index_associated = state_list.index(state)

        final_res_temp = 0

        for i in range(len(state_list)):
            final_res_temp =  self.tf.get_transition_values(state, p.get_action(state), hashmap[i])*dp_table[horizon-1][i]
        
        dp_table[horizon][index_associated] = self.r.get_reward(state, p.get_action(state))+final_res_temp  

        return dp_table[horizon][index_associated] 
    
