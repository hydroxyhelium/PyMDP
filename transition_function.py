## a wrapper for the transition function 
from typing import TypedDict
import numpy as np


## order is a mapping from state-> order in the map

class TF: 
    ## TF class can be initilized with transition matrix for each of the actions. 
    ## By default all the states would be from 0-(N-1). 
    ## You need to specify a mapping from your made up state to a number between 0, N-1
    def __init__(self, tf_matrix_hashmap: 'dict', order: 'dict') -> None:
        self.tf_matrix_hashmap = tf_matrix_hashmap
        self.order = order 
        return

    def get_transition_values(self, state1:'str', action: 'str', state2: 'str') -> int:
        transition_matrix_action = self.tf_matrix_hashmap[action]
        row = self.order[state1]
        column = self.order[state2]

        return transition_matrix_action[row][column]



    
