from lib.enviroment.multiAgentEnv.multiAgentTestEnv import multiAgentTestEnv
import numpy as np

class OneLevelTestEnv(multiAgentTestEnv):
    def __init__(self):
        super().__init__()

    def states(self):
        return super().states()
    
    def actions(self):
        return super().actions()
    
    def close(self):
        return super().close()
    
    def reset(self):
        return super().reset()
    
    def execute(self, actions):
        sampleTime,soc,remain,pricePreHour,hvacState1,hvacState2,hvacState3,intState,unIntState,intPreference,unintPreference,order = self.state

        self.intAgent.getState(self.totalState,self.interruptibleLoadActionMask)
        self.intAgent.environment.updateState(self.intAgent.states,self.interruptibleLoad)
        self.hvacAgent1.getState(self.totalState)
        self.hvacAgent1.environment.updateState(self.hvacAgent1.states)
        self.hvacAgent2.getState(self.totalState)
        self.hvacAgent2.environment.updateState(self.hvacAgent2.states)
        self.hvacAgent3.getState(self.totalState)
        self.hvacAgent3.environment.updateState(self.hvacAgent3.states)
        self.unIntAgent.getState(self.totalState,self.uninterruptibleLoadActionMask)
        self.unIntAgent.environment.updateState(self.unIntAgent.states,self.uninterruptibleLoad)
        self.socAgent.getState(self.totalState)
        self.socAgent.environment.updateState(self.socAgent.states)
        self.intAgent.execute()
        self.unIntAgent.execute()
        self.hvacAgent1.execute()
        self.hvacAgent2.execute()
        self.hvacAgent3.execute()
        self.socAgent.execute()

        self.updateTotalState()  

            
        self.state = self.stateAbstraction(self.totalState)
        done =  bool(sampleTime == 95)

        reward = 0
        states = dict(state=self.state)
        return states, done ,reward

    def stateAbstraction(self, totalState) -> np.array:
        return super().stateAbstraction(totalState)
    
    def updateTotalState(self):
        self.totalState["deltaSoc"] = self.socAgent.actions[0]
        self.totalState["SOC"]+=self.socAgent.actions[0]
        if self.totalState["SOC"]>=1 :
            self.totalState["SOC"]=1
        elif self.totalState["SOC"]<=0:
            self.totalState["SOC"]=0

        self.totalState["fixLoad"]+=self.hvacAgent1.actions[0]
        self.totalState["indoorTemperature1"] = self.hvacAgent1.states["state"][5]
        self.totalState["hvacPower1"] = self.hvacAgent1.actions[0]

        self.totalState["fixLoad"]+=self.hvacAgent2.actions[0]
        self.totalState["indoorTemperature2"] = self.hvacAgent2.states["state"][5]
        self.totalState["hvacPower2"] = self.hvacAgent2.actions[0]

        self.totalState["fixLoad"]+=self.hvacAgent3.actions[0]
        self.totalState["indoorTemperature3"] = self.hvacAgent3.states["state"][5]
        self.totalState["hvacPower3"] = self.hvacAgent3.actions[0]

        self.interruptibleLoad = self.intAgent.environment.interruptibleLoad
        self.interruptibleLoadActionMask = self.intAgent.states["action_mask"]
        self.totalState["intRemain"] = self.intAgent.states["state"][5]
        self.totalState["intSwitch"] = self.intAgent.actions
        if self.intAgent.actions==1:
            self.totalState["fixLoad"]+=self.interruptibleLoad.AvgPowerConsume
        
        self.uninterruptibleLoad = self.unIntAgent.environment.uninterruptibleLoad
        self.uninterruptibleLoadActionMask = self.unIntAgent.states["action_mask"]
        self.totalState["unintRemain"]=self.unIntAgent.states["state"][5]
        self.totalState["unintSwitch"]=self.unIntAgent.states["state"][6]
        if self.unIntAgent.states["state"][6]==1:
            self.totalState["fixLoad"]+=self.uninterruptibleLoad.AvgPowerConsume


        #Order = 0,1,2,3
        self.totalState["order"] = 0
        #update to next step
        if self.totalState["sampleTime"]!=95:
            self.totalState["sampleTime"]+=1
            self.totalState["fixLoad"]=self.Load[self.totalState["sampleTime"]]
            self.totalState["PV"]=self.PV[self.totalState["sampleTime"]]
            self.totalState["pricePerHour"]=self.GridPrice[self.totalState["sampleTime"]]
            self.totalState["outdoorTemperature"]=self.outdoorTemperature[self.totalState["sampleTime"]]
            self.totalState["userSetTemperature1"]=self.userSetTemperature1[self.totalState["sampleTime"]]
            self.totalState["userSetTemperature2"]=self.userSetTemperature2[self.totalState["sampleTime"]]
            self.totalState["userSetTemperature3"]=self.userSetTemperature3[self.totalState["sampleTime"]]
            self.totalState["unintRemain"]=self.unIntAgent.environment.uninterruptibleLoad.getRemainDemand()
            self.totalState["unintSwitch"]=self.unIntAgent.environment.uninterruptibleLoad.switch
            self.totalState["intPreference"] = self.intUserPreference[self.totalState["sampleTime"]]
            self.totalState["unintPreference"] = self.unintPreference[self.totalState["sampleTime"]]