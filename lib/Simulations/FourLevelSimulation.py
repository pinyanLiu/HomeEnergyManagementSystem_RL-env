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
        self.environment = Environment.create(environment = FourLevelTestEnv,max_episode_timesteps=672)
        self.agent = Agent.load(directory = 'HLA/FourLevelAgent/saver_dir',environment=self.environment)
        
    def simulation(self):
        return super().simulation()
    
    def EachMonthResult(self):
        for month in range(12):
            print("month ",month, " ExceedPgridMaxTimes: ",sum(self.testResult[month]["ExceedPgridMaxTimes"]))
            print("month ",month, " TotalHvacPreference: ",sum(self.testResult[month]["TotalHvacPreference"]))
            print("month ",month, " TotalIntPreference: ",sum(self.testResult[month]["TotalIntPreference"]))
            print("month ",month, " TotalUnintPreference: ",sum(self.testResult[month]["TotalUnintPreference"]))
            print("month ",month, " TotalElectricPrice: ",sum(self.testResult[month]["TotalElectricPrice"]))
            
    def outputResult(self):
        output = Plot(self.testResult)
        output.remainPower()
        output.soc()
        output.indoorTemperature(1)
        output.indoorTemperature(2)
        output.indoorTemperature(3)
        output.outdoorTemperature()
        output.userSetTemperature(1)
        output.userSetTemperature(2)
        output.userSetTemperature(3)
        output.price()
        output.plotIntPreference()
        output.plotUnintPreference()
        output.plotIntLoadPower()
        output.plotUnIntLoadPower()
        output.plotDeltaSOCPower()
        output.plotHVACPower(id=1)
        output.plotHVACPower(id=2)
        output.plotHVACPower(id=3)
        output.plotPVPower()
        output.plotPgridMax()
        #output.plotReward()
        output.plotResult('lib/plot/FourLevel/')    
    def __del__(self):
        return super().__del__()
    
