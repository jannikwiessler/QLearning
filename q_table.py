import numpy as np
from copy import deepcopy

class Qtable():
    """class returning a qtabel. 
    
    Examples:
    >>> qtable('fix_wrap_text_literals')
    'libfuturize.fixes.fix_wrap_text_lit
    """
    def __init__(self,max_observation, min_observation, descrete_os_size, number_of_actions):
        self.__max_observation = max_observation
        self.__min_observation = min_observation
        self.__descrete_os_size = descrete_os_size
        self.__action_space_dimension = number_of_actions

        self.__calc_os_win_size()
        self.__create_q_table()
        
    def __calc_os_win_size(self):
        self.__discrete_os_win_size = (self.__max_observation - self.__min_observation) / self.__descrete_os_size
    
    def __create_q_table(self):
        self.__q_table = np.random.uniform(low=-2, high=0,size=self.__descrete_os_size + [self.__action_space_dimension])

    def __calc_index(self, states):
        self.__index = (states - self.__min_observation) / self.__discrete_os_win_size
        self.__index = tuple(self.__index.astype(np.int))

    def to_index(self, states):
        self.__calc_index(states)
        return deepcopy(tuple(self.__index))

    def get_q_table(self):
        return deepcopy(self.__q_table)

    def get_action(self, states):
        self.__calc_index(states)
        return (np.argmax(self.__q_table[self.__index]))

    def get_max_q_value(self, states):
        self.__calc_index(states)
        return (np.max(self.__q_table[self.__index])) 

    def set_q_value(self, states, action, value = 0):
        self.__calc_index(states)
        self.__q_table[self.__index + (action, )] = value