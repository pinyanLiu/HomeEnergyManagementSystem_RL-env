from lib.enviroment.multiAgentEnv.multiAgentTrainEnv import multiAgentTrainEnv
from gym import make
import numpy as np
from random import randint,uniform
from lib.loads.interrupted import AC
from lib.loads.uninterrupted import WM

class multiAgentTestEnv(multiAgentTrainEnv):
    def __init__(self) :
        super().__init__()
        #each month pick one day for testing
        self.i = 0
        #import Load 
        self.allLoad = self.info.importTestingLoad()
        self.Load = self.allLoad.iloc[:,self.i].tolist()
        self.interruptibleLoad = AC(demand=self.intload_demand,AvgPowerConsume=self.intload_power)
        self.uninterruptibleLoad = WM(demand=self.unload_demand,executePeriod=self.unload_period,AvgPowerConsume=self.unload_power)



    def states(self):
        return super().states()

    def actions(self):
        return super().actions()
        
    def execute(self, actions):
        return super().execute(actions)

    def close(self):
        return super().close()

    def reset(self):
        '''
        Starting State
        '''
        self.i += 1
        self.Load = self.allLoad.iloc[:,self.i].tolist()
        self.interruptibleLoad.reset()
        self.uninterruptibleLoad.reset()

        #import PV
        if self.i  == 0:
            self.PV = self.allPV['Jan'].tolist()
            self.GridPrice = self.notSummerGridPrice
            self.outdoorTemperature = self.allOutdoorTemperature['Jan'].tolist()
            self.userSetTemperature = self.allUserSetTemperature['Jan'].tolist()
            self.intUserPreference = self.allIntPreference['1'].tolist()
            self.unintPreference = self.allUnintPreference['1'].tolist()

        elif self.i  == 1:
            self.PV = self.allPV['Feb'].tolist()
            self.GridPrice = self.notSummerGridPrice
            self.outdoorTemperature = self.allOutdoorTemperature['Feb'].tolist()
            self.userSetTemperature = self.allUserSetTemperature['Feb'].tolist()
            self.intUserPreference = self.allIntPreference['2'].tolist()
            self.unintPreference = self.allUnintPreference['2'].tolist()

        elif self.i  == 2:
            self.PV = self.allPV['Mar'].tolist()
            self.GridPrice = self.notSummerGridPrice
            self.outdoorTemperature = self.allOutdoorTemperature['Mar'].tolist()
            self.userSetTemperature = self.allUserSetTemperature['Mar'].tolist()
            self.intUserPreference = self.allIntPreference['3'].tolist()
            self.unintPreference = self.allUnintPreference['3'].tolist()

        elif self.i  == 3:
            self.PV = self.allPV['Apr'].tolist()
            self.GridPrice = self.notSummerGridPrice
            self.outdoorTemperature = self.allOutdoorTemperature['Apr'].tolist()
            self.userSetTemperature = self.allUserSetTemperature['Apr'].tolist()
            self.intUserPreference = self.allIntPreference['4'].tolist()
            self.unintPreference = self.allUnintPreference['4'].tolist()

        elif self.i  == 4:
            self.PV = self.allPV['May'].tolist()
            self.GridPrice = self.notSummerGridPrice
            self.outdoorTemperature = self.allOutdoorTemperature['May'].tolist()
            self.userSetTemperature = self.allUserSetTemperature['May'].tolist()
            self.intUserPreference = self.allIntPreference['5'].tolist()
            self.unintPreference = self.allUnintPreference['5'].tolist()

        elif self.i  == 5:
            self.PV = self.allPV['Jun'].tolist()
            self.GridPrice = self.summerGridPrice
            self.outdoorTemperature = self.allOutdoorTemperature['Jun'].tolist()
            self.userSetTemperature = self.allUserSetTemperature['Jun'].tolist()
            self.intUserPreference = self.allIntPreference['6'].tolist()
            self.unintPreference = self.allUnintPreference['6'].tolist()

        elif self.i  == 6:
            self.PV = self.allPV['July'].tolist()
            self.GridPrice = self.summerGridPrice
            self.outdoorTemperature = self.allOutdoorTemperature['July'].tolist()
            self.userSetTemperature = self.allUserSetTemperature['July'].tolist()
            self.intUserPreference = self.allIntPreference['7'].tolist()
            self.unintPreference = self.allUnintPreference['7'].tolist()

        elif self.i  == 7:
            self.PV = self.allPV['Aug'].tolist()
            self.GridPrice = self.summerGridPrice
            self.outdoorTemperature = self.allOutdoorTemperature['Aug'].tolist()
            self.userSetTemperature = self.allUserSetTemperature['Aug'].tolist()
            self.intUserPreference = self.allIntPreference['8'].tolist()
            self.unintPreference = self.allUnintPreference['8'].tolist()

        elif self.i  == 8:
            self.PV = self.allPV['Sep'].tolist()
            self.GridPrice = self.summerGridPrice
            self.outdoorTemperature = self.allOutdoorTemperature['Sep'].tolist()
            self.userSetTemperature = self.allUserSetTemperature['Sep'].tolist()
            self.intUserPreference = self.allIntPreference['9'].tolist()
            self.unintPreference = self.allUnintPreference['9'].tolist()

        elif self.i  == 9:
            self.PV = self.allPV['Oct'].tolist()
            self.GridPrice = self.notSummerGridPrice
            self.outdoorTemperature = self.allOutdoorTemperature['Oct'].tolist()
            self.userSetTemperature = self.allUserSetTemperature['Oct'].tolist()
            self.intUserPreference = self.allIntPreference['10'].tolist()
            self.unintPreference = self.allUnintPreference['10'].tolist()

        elif self.i  == 10:
            self.PV = self.allPV['Nov'].tolist()
            self.GridPrice = self.notSummerGridPrice
            self.outdoorTemperature = self.allOutdoorTemperature['Nov'].tolist()
            self.userSetTemperature = self.allUserSetTemperature['Nov'].tolist()
            self.intUserPreference = self.allIntPreference['11'].tolist()
            self.unintPreference = self.allUnintPreference['11'].tolist()

        elif self.i  == 11:
            self.PV = self.allPV['Dec'].tolist()
            self.GridPrice = self.notSummerGridPrice
            self.outdoorTemperature = self.allOutdoorTemperature['Dcb'].tolist()
            self.userSetTemperature = self.allUserSetTemperature['Dcb'].tolist()
            self.intUserPreference = self.allIntPreference['12'].tolist()
            self.unintPreference = self.allUnintPreference['12'].tolist()

        #reset state
        self.totalState = {
            "sampleTime":0,
            "fixLoad":self.Load[0],
            "PV":self.PV[0],
            "SOC":self.socInit,
            "pricePerHour":self.GridPrice[0],
            "deltaSoc":0,
            "indoorTemperature":self.initIndoorTemperature,
            "outdoorTemperature":self.outdoorTemperature[0],
            "userSetTemperature":self.userSetTemperature[0],
            "intRemain":self.interruptibleLoad.demand,
            "intSwitch":self.interruptibleLoad.switch,
            "intPreference":self.intUserPreference[0],
            "unintRemain":self.uninterruptibleLoad.demand,
            "unintSwitch":self.uninterruptibleLoad.switch,
            "unintPreference":self.unintPreference[0],
            "order":0
        }
        self.interruptibleLoadActionMask = [True,True]
        self.uninterruptibleLoadActionMask = [True,True]
        self.action_mask = [True,True,True,True,True]
        self.state = self.stateAbstraction(self.totalState)
        self.socAgent.agent.internals = self.socAgent.agent.initial_internals()
        self.hvacAgent.agent.internals = self.hvacAgent.agent.initial_internals()
        self.intAgent.agent.internals = self.intAgent.agent.initial_internals()
        self.unIntAgent.agent.internals = self.unIntAgent.agent.initial_internals()
        self.socAgent.environment.reset()
        self.hvacAgent.environment.reset()
        self.intAgent.environment.reset()
        self.unIntAgent.environment.reset()
        
        return self.state


    def stateAbstraction(self, totalState) -> np.array:
        return super().stateAbstraction(totalState)
    
    def updateTotalState(self, mode):
        return super().updateTotalState(mode)