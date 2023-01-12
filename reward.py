## Reward function will be wrapped with this class

class Reward:
    def __init__(self, rf_dict:dict)->None:
        self.rf_dict=rf_dict
        return

    def get_reward(self, state:str, action;str)->int:
        temp_tuple = (state, action)
        return self.rf_dict[temp_tuple]
        
    
