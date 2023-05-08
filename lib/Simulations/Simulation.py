import pandas as pd
import numpy as np

class Simulation():
    def __init__(self) -> None:
        self.testResult = {}
        for month in range(12):
            self.testResult[month] = pd.DataFrame()
        self.totalReward = []
    
    def simulation(self):
        print("do simulation")
    
    def outputResult(self):
        print("plot result")

        

    def getMean(self):
        return(np.mean(self.totalReward))

    def getStd(self):
        return(np.std(self.totalReward))

    def getMin(self):
        return(np.min(self.totalReward))
    def getMax(self):
        return(np.max(self.totalReward))
        
    def __del__(self):
        # Close agent and environment
        pass