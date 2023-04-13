from lib.Simulations.multiSimulation import multiSimulation
from tensorforce import Agent,Environment

from lib.enviroment.multiAgentEnv.FourLevelTestEnv import FourLevelTestEnv
import pandas as pd
class FourLevelSimulation(multiSimulation):
    def __init__(self):
        self.testResult = {}
        for month in range(12):
            self.testResult[month] = pd.DataFrame()
        self.totalReward = []        
        self.environment = Environment.create(environment = FourLevelTestEnv,max_episode_timesteps=384)
        self.agent = Agent.load(directory = 'HLA/saver_dir',environment=self.environment)
    def simulation(self):
        return super().simulation()
    
    def outputResult(self):
        return super().outputResult()
    
    def __del__(self):
        return super().__del__()
    
