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
        self.interruptibleLoad1 = AC(demand=self.intload_demand1,AvgPowerConsume=self.intload_power1)
        self.interruptibleLoad2 = AC(demand=self.intload_demand2,AvgPowerConsume=self.intload_power2)
        self.interruptibleLoad3 = AC(demand=self.intload_demand3,AvgPowerConsume=self.intload_power3)
        self.uninterruptibleLoad1 = WM(demand=self.unload_demand1,executePeriod=self.unload_period1,AvgPowerConsume=self.unload_power1)
        self.uninterruptibleLoad2 = WM(demand=self.unload_demand2,executePeriod=self.unload_period2,AvgPowerConsume=self.unload_power2)



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
        self.interruptibleLoad1.reset()
        self.interruptibleLoad2.reset()
        self.interruptibleLoad3.reset()
        self.uninterruptibleLoad1.reset()
        self.uninterruptibleLoad2.reset()

        #import PV
        if self.i  == 0:
            self.PV = self.allPV['Jan'].tolist()
            self.GridPrice = self.notSummerGridPrice
            self.outdoorTemperature = self.allOutdoorTemperature['Jan'].tolist()
            self.userSetTemperature1 = self.allUserSetTemperature1['Jan'].tolist()
            self.userSetTemperature2 = self.allUserSetTemperature2['Jan'].tolist()
            self.userSetTemperature3 = self.allUserSetTemperature3['Jan'].tolist()
            self.intUserPreference1 = self.allIntPreference1['1'].tolist()
            self.intUserPreference2 = self.allIntPreference2['1'].tolist()
            self.intUserPreference3 = self.allIntPreference3['1'].tolist()
            self.unintPreference1 = self.allUnintPreference1['1'].tolist()
            self.unintPreference2 = self.allUnintPreference2['1'].tolist()

        elif self.i  == 1:
            self.PV = self.allPV['Feb'].tolist()
            self.GridPrice = self.notSummerGridPrice
            self.outdoorTemperature = self.allOutdoorTemperature['Feb'].tolist()
            self.userSetTemperature1 = self.allUserSetTemperature1['Feb'].tolist()
            self.userSetTemperature2 = self.allUserSetTemperature2['Feb'].tolist()
            self.userSetTemperature3 = self.allUserSetTemperature3['Feb'].tolist()
            self.intUserPreference1 = self.allIntPreference1['2'].tolist()
            self.intUserPreference2 = self.allIntPreference2['2'].tolist()
            self.intUserPreference3 = self.allIntPreference3['2'].tolist()
            self.unintPreference1 = self.allUnintPreference1['2'].tolist()
            self.unintPreference2 = self.allUnintPreference2['2'].tolist()

        elif self.i  == 2:
            self.PV = self.allPV['Mar'].tolist()
            self.GridPrice = self.notSummerGridPrice
            self.outdoorTemperature = self.allOutdoorTemperature['Mar'].tolist()
            self.userSetTemperature1 = self.allUserSetTemperature1['Mar'].tolist()
            self.userSetTemperature2 = self.allUserSetTemperature2['Mar'].tolist()
            self.userSetTemperature3 = self.allUserSetTemperature3['Mar'].tolist()
            self.intUserPreference1 = self.allIntPreference1['3'].tolist()
            self.intUserPreference2 = self.allIntPreference2['3'].tolist()
            self.intUserPreference3 = self.allIntPreference3['3'].tolist()
            self.unintPreference1 = self.allUnintPreference1['3'].tolist()
            self.unintPreference2 = self.allUnintPreference2['3'].tolist()

        elif self.i  == 3:
            self.PV = self.allPV['Apr'].tolist()
            self.GridPrice = self.notSummerGridPrice
            self.outdoorTemperature = self.allOutdoorTemperature['Apr'].tolist()
            self.userSetTemperature1 = self.allUserSetTemperature1['Apr'].tolist()
            self.userSetTemperature2 = self.allUserSetTemperature2['Apr'].tolist()
            self.userSetTemperature3 = self.allUserSetTemperature3['Apr'].tolist()
            self.intUserPreference1 = self.allIntPreference1['4'].tolist()
            self.intUserPreference2 = self.allIntPreference2['4'].tolist()
            self.intUserPreference3 = self.allIntPreference3['4'].tolist()
            self.unintPreference1 = self.allUnintPreference1['4'].tolist()
            self.unintPreference2 = self.allUnintPreference2['4'].tolist()

        elif self.i  == 4:
            self.PV = self.allPV['May'].tolist()
            self.GridPrice = self.notSummerGridPrice
            self.outdoorTemperature = self.allOutdoorTemperature['May'].tolist()
            self.userSetTemperature1 = self.allUserSetTemperature1['May'].tolist()
            self.userSetTemperature2 = self.allUserSetTemperature2['May'].tolist()
            self.userSetTemperature3 = self.allUserSetTemperature3['May'].tolist()
            self.intUserPreference1 = self.allIntPreference1['5'].tolist()
            self.intUserPreference2 = self.allIntPreference2['5'].tolist()
            self.intUserPreference3 = self.allIntPreference3['5'].tolist()
            self.unintPreference1 = self.allUnintPreference1['5'].tolist()
            self.unintPreference2 = self.allUnintPreference2['5'].tolist()

        elif self.i  == 5:
            self.PV = self.allPV['Jun'].tolist()
            self.GridPrice = self.summerGridPrice
            self.outdoorTemperature = self.allOutdoorTemperature['Jun'].tolist()
            self.userSetTemperature1 = self.allUserSetTemperature1['Jun'].tolist()
            self.userSetTemperature2 = self.allUserSetTemperature2['Jun'].tolist()
            self.userSetTemperature3 = self.allUserSetTemperature3['Jun'].tolist()
            self.intUserPreference1 = self.allIntPreference1['6'].tolist()
            self.intUserPreference2 = self.allIntPreference2['6'].tolist()
            self.intUserPreference3 = self.allIntPreference3['6'].tolist()
            self.unintPreference1 = self.allUnintPreference1['6'].tolist()
            self.unintPreference2 = self.allUnintPreference2['6'].tolist()

        elif self.i  == 6:
            self.PV = self.allPV['July'].tolist()
            self.GridPrice = self.summerGridPrice
            self.outdoorTemperature = self.allOutdoorTemperature['July'].tolist()
            self.userSetTemperature1 = self.allUserSetTemperature1['July'].tolist()
            self.userSetTemperature2 = self.allUserSetTemperature2['July'].tolist()
            self.userSetTemperature3 = self.allUserSetTemperature3['July'].tolist()
            self.intUserPreference1 = self.allIntPreference1['7'].tolist()
            self.intUserPreference2 = self.allIntPreference2['7'].tolist()
            self.intUserPreference3 = self.allIntPreference3['7'].tolist()
            self.unintPreference1 = self.allUnintPreference1['7'].tolist()
            self.unintPreference2 = self.allUnintPreference2['7'].tolist()

        elif self.i  == 7:
            self.PV = self.allPV['Aug'].tolist()
            self.GridPrice = self.summerGridPrice
            self.outdoorTemperature = self.allOutdoorTemperature['Aug'].tolist()
            self.userSetTemperature1 = self.allUserSetTemperature1['Aug'].tolist()
            self.userSetTemperature2 = self.allUserSetTemperature2['Aug'].tolist()
            self.userSetTemperature3 = self.allUserSetTemperature3['Aug'].tolist()
            self.intUserPreference1 = self.allIntPreference1['8'].tolist()
            self.intUserPreference2 = self.allIntPreference2['8'].tolist()
            self.intUserPreference3 = self.allIntPreference3['8'].tolist()
            self.unintPreference1 = self.allUnintPreference1['8'].tolist()
            self.unintPreference2 = self.allUnintPreference2['8'].tolist()

        elif self.i  == 8:
            self.PV = self.allPV['Sep'].tolist()
            self.GridPrice = self.summerGridPrice
            self.outdoorTemperature = self.allOutdoorTemperature['Sep'].tolist()
            self.userSetTemperature1 = self.allUserSetTemperature1['Sep'].tolist()
            self.userSetTemperature2 = self.allUserSetTemperature2['Sep'].tolist()
            self.userSetTemperature3 = self.allUserSetTemperature3['Sep'].tolist()
            self.intUserPreference1 = self.allIntPreference1['9'].tolist()
            self.intUserPreference2 = self.allIntPreference2['9'].tolist()
            self.intUserPreference3 = self.allIntPreference3['9'].tolist()
            self.unintPreference1 = self.allUnintPreference1['9'].tolist()
            self.unintPreference2 = self.allUnintPreference2['9'].tolist()

        elif self.i  == 9:
            self.PV = self.allPV['Oct'].tolist()
            self.GridPrice = self.notSummerGridPrice
            self.outdoorTemperature = self.allOutdoorTemperature['Oct'].tolist()
            self.userSetTemperature1 = self.allUserSetTemperature1['Oct'].tolist()
            self.userSetTemperature2 = self.allUserSetTemperature2['Oct'].tolist()
            self.userSetTemperature3 = self.allUserSetTemperature3['Oct'].tolist()
            self.intUserPreference1 = self.allIntPreference1['10'].tolist()
            self.intUserPreference2 = self.allIntPreference2['10'].tolist()
            self.intUserPreference3 = self.allIntPreference3['10'].tolist()
            self.unintPreference1 = self.allUnintPreference1['10'].tolist()
            self.unintPreference2 = self.allUnintPreference2['10'].tolist()

        elif self.i  == 10:
            self.PV = self.allPV['Nov'].tolist()
            self.GridPrice = self.notSummerGridPrice
            self.outdoorTemperature = self.allOutdoorTemperature['Nov'].tolist()
            self.userSetTemperature1 = self.allUserSetTemperature1['Nov'].tolist()
            self.userSetTemperature2 = self.allUserSetTemperature2['Nov'].tolist()
            self.userSetTemperature3 = self.allUserSetTemperature3['Nov'].tolist()
            self.intUserPreference1 = self.allIntPreference1['11'].tolist()
            self.intUserPreference2 = self.allIntPreference2['11'].tolist()
            self.intUserPreference3 = self.allIntPreference3['11'].tolist()
            self.unintPreference1 = self.allUnintPreference1['11'].tolist()
            self.unintPreference2 = self.allUnintPreference2['11'].tolist()

        elif self.i  == 11:
            self.PV = self.allPV['Dec'].tolist()
            self.GridPrice = self.notSummerGridPrice
            self.outdoorTemperature = self.allOutdoorTemperature['Dcb'].tolist()
            self.userSetTemperature1 = self.allUserSetTemperature1['Dcb'].tolist()
            self.userSetTemperature2 = self.allUserSetTemperature2['Dcb'].tolist()
            self.userSetTemperature3 = self.allUserSetTemperature3['Dcb'].tolist()
            self.intUserPreference1 = self.allIntPreference1['12'].tolist()
            self.intUserPreference2 = self.allIntPreference2['12'].tolist()
            self.intUserPreference3 = self.allIntPreference3['12'].tolist()
            self.unintPreference1 = self.allUnintPreference1['12'].tolist()
            self.unintPreference2 = self.allUnintPreference2['12'].tolist()

        #reset state
        self.totalState = {
            "sampleTime":0,
            "fixLoad":self.Load[0],
            "PV":self.PV[0],
            "SOC":self.socInit,
            "pricePerHour":self.GridPrice[0],
            "deltaSoc":0,
            "indoorTemperature1":self.initIndoorTemperature,
            "indoorTemperature2":self.initIndoorTemperature,
            "indoorTemperature3":self.initIndoorTemperature,
            "outdoorTemperature":self.outdoorTemperature[0],
            "userSetTemperature1":self.userSetTemperature1[0],
            "userSetTemperature2":self.userSetTemperature2[0],
            "userSetTemperature3":self.userSetTemperature3[0],
            "hvacPower1":0,
            "hvacPower2":0,
            "hvacPower3":0,
            "intRemain1":self.interruptibleLoad1.demand,
            "intSwitch1":self.interruptibleLoad1.switch,
            "intPreference1":self.intUserPreference1[0],
            "intRemain2":self.interruptibleLoad2.demand,
            "intSwitch2":self.interruptibleLoad2.switch,
            "intPreference2":self.intUserPreference2[0],
            "intRemain3":self.interruptibleLoad3.demand,
            "intSwitch3":self.interruptibleLoad3.switch,
            "intPreference3":self.intUserPreference3[0],
            "unintRemain1":self.uninterruptibleLoad1.demand*self.uninterruptibleLoad1.executePeriod,
            "unintSwitch1":self.uninterruptibleLoad1.switch,
            "unintPreference1":self.unintPreference1[0],
            "unintRemain2":self.uninterruptibleLoad2.demand*self.uninterruptibleLoad2.executePeriod,
            "unintSwitch2":self.uninterruptibleLoad2.switch,
            "unintPreference2":self.unintPreference2[0],
            "order":0,
            "PgridMax":self.PgridMax
        }
        self.interruptibleLoadActionMask1 = [True,True]
        self.interruptibleLoadActionMask2 = [True,True]
        self.interruptibleLoadActionMask3 = [True,True]
        self.uninterruptibleLoadActionMask1 = [True,True]
        self.uninterruptibleLoadActionMask2 = [True,True]
        self.action_mask = [True,True,True,True,True,True,True,True,True,True]
        self.state = self.stateAbstraction(self.totalState)
        self.socAgent.agent.internals = self.socAgent.agent.initial_internals()
        self.hvacAgent1.agent.internals = self.hvacAgent1.agent.initial_internals()
        self.hvacAgent2.agent.internals = self.hvacAgent2.agent.initial_internals()
        self.hvacAgent3.agent.internals = self.hvacAgent3.agent.initial_internals()
        self.intAgent1.agent.internals = self.intAgent1.agent.initial_internals()
        self.intAgent2.agent.internals = self.intAgent2.agent.initial_internals()
        self.intAgent3.agent.internals = self.intAgent3.agent.initial_internals()
        self.unIntAgent1.agent.internals = self.unIntAgent1.agent.initial_internals()
        self.unIntAgent2.agent.internals = self.unIntAgent2.agent.initial_internals()
        self.socAgent.environment.reset()
        self.hvacAgent1.environment.reset()
        self.hvacAgent2.environment.reset()
        self.hvacAgent3.environment.reset()
        self.intAgent1.environment.reset()
        self.intAgent2.environment.reset()
        self.intAgent3.environment.reset()
        self.unIntAgent1.environment.reset()
        self.unIntAgent2.environment.reset()

        
        return self.state


    def stateAbstraction(self, totalState) -> np.array:
        return super().stateAbstraction(totalState)
    
    def updateTotalState(self, mode):
        return super().updateTotalState(mode)