from MDP import MDP
from reward import Reward
from transition_function import TF
from policy import Policy

## Consider an MDP with states (0, 1, 2, 3)

states = ['0', '1', '2', '3']
actions = ['b', 'c']

matrix_b = [[0, 0.9, 0.1, 0.0], [0.9, 0.1, 0.0, 0.0], [0.0, 0.0, 0.1, 0.9], [0.9, 0.0, 0.0, 0.1]]
matrix_c = [[0, 0.1, 0.9, 0.0], [0.9, 0.1, 0.0, 0.0], [0.0, 0.0, 0.1, 0.9], [0.9, 0.0, 0.0, 0.1]]

tf_matrix_hashmap = {'b': matrix_b, 'c':matrix_c}
order = {'0':0, '1':1, '2': 2, '3':3}


tf = TF(tf_matrix_hashmap,order)


## contains mapping from (state, action) to reward
rf_dict = {('0', 'b'):0, ('0', 'c'):0, ('1', 'b'):1, ('1', 'c'):1, ('2', 'b'):0, ('2', 'c'):0, ('3', 'b'):2, ('3', 'c'):2}

r = Reward(rf_dict)

## checks if the transition matrix works correctly 
print(tf.get_transition_values('0', 'b', '2'))

print(r.get_reward('0','c'))

## checking if the constructor functions for MDP works perfectly or not 
mdp = MDP(states, actions, tf, '0', r, 1)

policy_mapping = {'0':'c', '1':'c', '2':'c', '3':'c'}

p = Policy(policy_mapping)

print(mdp.evaluate_policy_finite(p,2,'1'))
## this should print 1.1 which it does. 







