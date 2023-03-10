from lib.loads.uninterrupted import WM
from gym import make
import numpy as np
from lib.enviroment.UnInterruptibleLoadTrainEnv import UnIntEnv

class UnIntTest(UnIntEnv):
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