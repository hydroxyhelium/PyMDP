## this is a class used to describe Markov Descision Process (MPD)  which forms the backbone for RNN and 
## Reinforcement Learning
from transition_function import TF
from reward import Reward
from policy import Policy
import random

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
        
    def evaluate_policy_finite(self, ):
    
