from lib.Simulations.multiSimulation import multiSimulation
from tensorforce import Agent,Environment
from lib.Simulations.Simulation import Simulation
from lib.plot.plot import Plot

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
        output = Plot(self.testResult)
        output.remainPower()
        output.soc()
        output.indoorTemperature()
        output.outdoorTemperature()
        output.userSetTemperature()
        output.price()
        output.plotIntPreference()
        output.plotUnintPreference()
        output.plotIntLoadPower()
        output.plotUnIntLoadPower()
        output.plotDeltaSOCPower()
        output.plotReward()
        output.plotResult('lib/plot/FourLevel/')    
    def __del__(self):
        return super().__del__()
    
