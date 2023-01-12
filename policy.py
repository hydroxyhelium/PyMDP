## a class wrapper for a policy

class Policy:
    ## mapping is a dict that contains mapping from various states to action
    ## the agent would take in that particular state
    def __init__(self, mapping:dict) -> None:
        self.mapping = mapping
    
    ## return the string that user would get back
    def get_action(self, state: str) -> str:
        return self.mapping[state]


    
