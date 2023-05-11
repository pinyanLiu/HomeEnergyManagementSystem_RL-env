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

    def EachMonthResult(self):
        for month in range(12):
            print("month ",month, " ExceedPgridMaxTimes: ",sum(self.testResult[month]["ExceedPgridMaxTimes"]))
            print("month ",month, " TotalHvacPreference: ",sum(self.testResult[month]["TotalHvacPreference"]))
            print("month ",month, " TotalIntPreference: ",sum(self.testResult[month]["TotalIntPreference"]))
            print("month ",month, " TotalUnintPreference: ",sum(self.testResult[month]["TotalUnintPreference"]))
            print("month ",month, " TotalElectricPrice: ",sum(self.testResult[month]["TotalElectricPrice"]))
            print("month ",month, " intRemain1: ",self.testResult[month]["intRemain1"].iloc[95])
            print("month ",month, " intRemain2: ",self.testResult[month]["intRemain2"].iloc[95])
            print("month ",month, " intRemain3: ",self.testResult[month]["intRemain3"].iloc[95])
            print("month ",month, " unloadRemain1: ",self.testResult[month]["unloadRemain1"].iloc[95])
            print("month ",month, " unloadRemain2: ",self.testResult[month]["unloadRemain2"].iloc[95])
            print("month ",month, " soc: ",self.testResult[month]["soc"].iloc[95])

    def avgMonthResult(self):
        ExceedPgridMaxTimes = 0 
        TotalHvacPreference = 0
        TotalIntPreference = 0
        TotalUnintPreference = 0
        TotalElectricPrice = 0
        for month in range(12):
            ExceedPgridMaxTimes += sum(self.testResult[month]["ExceedPgridMaxTimes"])
            TotalHvacPreference += sum(self.testResult[month]["TotalHvacPreference"])
            TotalIntPreference  += sum(self.testResult[month]["TotalIntPreference"])
            TotalUnintPreference += sum(self.testResult[month]["TotalUnintPreference"])
            TotalElectricPrice += sum(self.testResult[month]["TotalElectricPrice"])
        print("Exceed PgridMax Ratio :", ExceedPgridMaxTimes/(12*96)*100,"%")
        print("AVG TotalHvacPreference :", TotalHvacPreference/12)
        print("AVG TotalIntPreference :", TotalIntPreference/12)
        print("AVG TotalUnintPreference:", TotalUnintPreference/12)
        print("AVG TotalElectricPrice:", TotalElectricPrice/12)


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