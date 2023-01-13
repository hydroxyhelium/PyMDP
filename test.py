from MDP import MDP
from reward import Reward
from transition_function import TF


## Consider an MDP with states (0, 1, 2, 3)

states = {'0', '1', '2', '3'}
actions = {'b', 'c'}

matrix_b = [[0, 0.9, 1.0, 0.0], [0.9, 0.1, 0.0, 0.0], [0.0, 0.0, 0.1, 0.9], [0.9, 0.0, 0.0, 0.1]]
matrix_c = [[0, 0.1, 0.9, 0.0], [0.9, 0.1, 0.0, 0.0], [0.0, 0.0, 0.1, 0.9], [0.9, 0.0, 0.0, 0.1]]

tf_matrix_hashmap = {'b': matrix_b, 'c':matrix_c}
order = {'0':0, '1':1, '2': 2, '3':3}









