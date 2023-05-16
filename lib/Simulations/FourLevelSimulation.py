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
        self.environment = Environment.create(environment = FourLevelTestEnv,max_episode_timesteps=960)
        self.agent = Agent.load(directory = 'HLA/FourLevelAgent/saver_dir',environment=self.environment)
        
    def simulation(self):
        return super().simulation()
    
    def EachMonthResult(self):
        return super().EachMonthResult()

    def avgMonthResult(self):
        return super().avgMonthResult()
            
    def outputResult(self,mode):
        output = Plot(self.testResult)
        if mode == "hvac":
            output.remainPower()
            output.soc()
            output.indoorTemperature(1)
            output.indoorTemperature(2)
            output.indoorTemperature(3)
            output.outdoorTemperature()
            output.userSetTemperature(1)
            output.userSetTemperature(2)
            output.userSetTemperature(3)
            output.plotHVACPower(id=1)
            output.plotHVACPower(id=2)
            output.plotHVACPower(id=3)
            output.plotDeltaSOCPower()
            output.price()
            output.plotPgridMax()
            output.plotPVPower()
            output.plotResult('lib/plot/FourLevel/hvac/')

        elif mode == "soc":
            output.remainPower()
            output.soc()
            output.price()
            output.plotPgridMax()
            output.plotPVPower()
            output.plotDeltaSOCPower()
            output.plotResult('lib/plot/FourLevel/soc/')
            
        elif mode == "int":
            output.remainPower()
            output.soc()
            output.price()
            output.plotPgridMax()
            output.plotPVPower()
            output.plotIntPreference(1)
            output.plotIntPreference(2)
            output.plotIntPreference(3)
            output.plotIntLoadPower(1)
            output.plotIntLoadPower(2)
            output.plotIntLoadPower(3)
            output.plotDeltaSOCPower()
            output.plotResult('lib/plot/FourLevel/int/')

        elif mode == "unint":
            output.remainPower()
            output.soc()
            output.price()
            output.plotPgridMax()
            output.plotPVPower()
            output.plotUnintPreference(1)
            output.plotUnintPreference(2)
            output.plotUnIntLoadPower(1)
            output.plotUnIntLoadPower(2)
            output.plotResult('lib/plot/FourLevel/unint/')
        else:
            output.remainPower()
            output.soc()
            output.price()
            output.plotPgridMax()
            output.plotPVPower()
            output.plotDeltaSOCPower()
            output.plotHVACPower(id=1)
            output.plotHVACPower(id=2)
            output.plotHVACPower(id=3)
            output.plotIntPreference(1)
            output.plotIntPreference(2)
            output.plotIntPreference(3)
            output.plotIntLoadPower(1)
            output.plotIntLoadPower(2)
            output.plotIntLoadPower(3)
            output.plotUnintPreference(1)
            output.plotUnintPreference(2)
            output.plotUnIntLoadPower(1)
            output.plotUnIntLoadPower(2)
            output.plotResult('lib/plot/FourLevel/')

    def __del__(self):
        return super().__del__()
    
