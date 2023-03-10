from lib.enviroment.HVACTrainEnv import HvacEnv
import numpy as np

class HvacTest(HvacEnv):
    def __init__(self) :
        pass

    def states(self):
        return super().states()

    def actions(self):
        return super().actions()
        
    def execute(self, actions):
        return super().execute(actions)
    
    def reset(self):
        pass