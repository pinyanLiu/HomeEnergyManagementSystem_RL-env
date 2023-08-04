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
        sampleTime,soc,remain,pricePreHour,hvacState1,hvacState2,hvacState3,intState1,intState2,intState3,unIntState1,unIntState2,intPreference1,intPreference2,intPreference3,unintPreference1,unintPreference2,order = self.state
        if order == 0:
            self.socAgent.getState(self.totalState)
            self.socAgent.environment.updateState(self.socAgent.states)
            self.intAgent1.getState(self.totalState,self.interruptibleLoadActionMask1)
            self.intAgent1.environment.updateState(self.intAgent1.states,self.interruptibleLoad1)
            self.intAgent2.getState(self.totalState,self.interruptibleLoadActionMask2)
            self.intAgent2.environment.updateState(self.intAgent2.states,self.interruptibleLoad2)
            self.intAgent3.getState(self.totalState,self.interruptibleLoadActionMask3)
            self.intAgent3.environment.updateState(self.intAgent3.states,self.interruptibleLoad3)
            self.hvacAgent1.getState(self.totalState)
            self.hvacAgent1.environment.updateState(self.hvacAgent1.states)
            self.hvacAgent2.getState(self.totalState)
            self.hvacAgent2.environment.updateState(self.hvacAgent2.states)
            self.hvacAgent3.getState(self.totalState)
            self.hvacAgent3.environment.updateState(self.hvacAgent3.states)
            self.unIntAgent1.getState(self.totalState,self.uninterruptibleLoadActionMask1)
            self.unIntAgent1.environment.updateState(self.unIntAgent1.states,self.uninterruptibleLoad1)
            self.unIntAgent2.getState(self.totalState,self.uninterruptibleLoadActionMask2)
            self.unIntAgent2.environment.updateState(self.unIntAgent2.states,self.uninterruptibleLoad2)
            self.intAgent1.execute()
            self.intAgent2.execute()
            self.intAgent3.execute()
            self.unIntAgent1.execute()
            self.unIntAgent2.execute()
            self.hvacAgent1.execute()
            self.hvacAgent2.execute()
            self.hvacAgent3.execute()
            self.socAgent.execute()
        self.updateTotalState()  

        

            
        self.state = self.stateAbstraction(self.totalState)
        done =  bool(sampleTime == 95 and order ==1)

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

        self.interruptibleLoad1 = self.intAgent1.environment.interruptibleLoad
        self.interruptibleLoadActionMask1 = self.intAgent1.states["action_mask"]
        self.totalState["intRemain1"] = self.intAgent1.states["state"][5]
        self.totalState["intSwitch1"] = self.intAgent1.actions
        if self.intAgent1.actions==1:
            self.totalState["fixLoad"]+=self.interruptibleLoad1.AvgPowerConsume

        self.interruptibleLoad2 = self.intAgent2.environment.interruptibleLoad
        self.interruptibleLoadActionMask2 = self.intAgent2.states["action_mask"]
        self.totalState["intRemain2"] = self.intAgent2.states["state"][5]
        self.totalState["intSwitch2"] = self.intAgent2.actions
        if self.intAgent2.actions==1:
            self.totalState["fixLoad"]+=self.interruptibleLoad2.AvgPowerConsume

        self.interruptibleLoad3 = self.intAgent3.environment.interruptibleLoad
        self.interruptibleLoadActionMask3 = self.intAgent3.states["action_mask"]
        self.totalState["intRemain3"] = self.intAgent3.states["state"][5]
        self.totalState["intSwitch3"] = self.intAgent3.actions
        if self.intAgent3.actions==1:
            self.totalState["fixLoad"]+=self.interruptibleLoad3.AvgPowerConsume
        
        self.uninterruptibleLoad1 = self.unIntAgent1.environment.uninterruptibleLoad
        self.uninterruptibleLoadActionMask1 = self.unIntAgent1.states["action_mask"]
        self.totalState["unintRemain1"]=self.unIntAgent1.states["state"][5]
        self.totalState["unintSwitch1"]=self.unIntAgent1.states["state"][6]
        if self.unIntAgent1.states["state"][6]==1:
            self.totalState["fixLoad"]+=self.uninterruptibleLoad1.AvgPowerConsume

        self.uninterruptibleLoad2 = self.unIntAgent2.environment.uninterruptibleLoad
        self.uninterruptibleLoadActionMask2 = self.unIntAgent2.states["action_mask"]
        self.totalState["unintRemain"]=self.unIntAgent2.states["state"][5]
        self.totalState["unintSwitch"]=self.unIntAgent2.states["state"][6]
        if self.unIntAgent2.states["state"][6]==1:
            self.totalState["fixLoad"]+=self.uninterruptibleLoad2.AvgPowerConsume

        #Order = 0,1
        self.totalState["order"] = (self.totalState["order"]+1 if self.totalState["order"]<1 else 0 )
        #update to next step
        if self.totalState["order"] == 0 and self.totalState["sampleTime"]!=95:
            self.totalState["sampleTime"]+=1
            self.totalState["fixLoad"]=self.Load[self.totalState["sampleTime"]]
            self.totalState["PV"]=self.PV[self.totalState["sampleTime"]]
            self.totalState["pricePerHour"]=self.GridPrice[self.totalState["sampleTime"]]
            self.totalState["deltaSoc"] = 0
            self.totalState["outdoorTemperature"]=self.outdoorTemperature[self.totalState["sampleTime"]]
            self.totalState["userSetTemperature1"]=self.userSetTemperature1[self.totalState["sampleTime"]]
            self.totalState["userSetTemperature2"]=self.userSetTemperature2[self.totalState["sampleTime"]]
            self.totalState["userSetTemperature3"]=self.userSetTemperature3[self.totalState["sampleTime"]]
            self.totalState["unintRemain1"]=self.unIntAgent1.environment.uninterruptibleLoad.getRemainDemand()
            self.totalState["unintRemain2"]=self.unIntAgent2.environment.uninterruptibleLoad.getRemainDemand()
            self.totalState["unintSwitch1"]=self.unIntAgent1.environment.uninterruptibleLoad.switch
            self.totalState["unintSwitch2"]=self.unIntAgent2.environment.uninterruptibleLoad.switch
            self.totalState["intSwitch1"] = 0
            self.totalState["intSwitch2"] = 0
            self.totalState["intSwitch3"] = 0
            self.totalState["intPreference1"] = self.intUserPreference1[self.totalState["sampleTime"]]
            self.totalState["intPreference2"] = self.intUserPreference2[self.totalState["sampleTime"]]
            self.totalState["intPreference3"] = self.intUserPreference3[self.totalState["sampleTime"]]
            self.totalState["unintPreference1"] = self.unintPreference1[self.totalState["sampleTime"]]
            self.totalState["unintPreference2"] = self.unintPreference2[self.totalState["sampleTime"]]