from lib.enviroment.multiAgentEnv.multiAgentTestEnv import multiAgentTestEnv
import numpy as np

class FourLevelTestEnv(multiAgentTestEnv):
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
        sampleTime,soc,remain,pricePreHour,hvacState1,hvacState2,hvacState3,intState1,intState2,intState3,unIntState1,unIntState2,intPreference1,intPreference2,intPreference3,unintPreference1,unintPreference2,order = self.state    #execute HVAC
        if order == 1:
            self.hvacAgent1.getState(self.totalState)
            self.hvacAgent1.environment.updateState(self.hvacAgent1.states)
            self.hvacAgent1.execute()
            # reward.append(self.hvacAgent1.reward)
            self.updateTotalState("hvac1")
        
        elif order ==2:
            self.hvacAgent2.getState(self.totalState)
            self.hvacAgent2.environment.updateState(self.hvacAgent2.states)
            self.hvacAgent2.execute()
            # reward.append(self.hvacAgent2.reward)
            self.updateTotalState("hvac2")

        elif order ==3:
            self.hvacAgent3.getState(self.totalState)
            self.hvacAgent3.environment.updateState(self.hvacAgent3.states)
            self.hvacAgent3.execute()
            # reward.append(self.hvacAgent3.reward)
            self.updateTotalState("hvac3")
        
    #execute interruptible load
        elif order == 6:
            self.intAgent1.getState(self.totalState,self.interruptibleLoadActionMask1)
            self.intAgent1.environment.updateState(self.intAgent1.states,self.interruptibleLoad1)
            self.intAgent1.execute()
            # reward.append(self.intAgent.reward)
            self.updateTotalState("int1")   

    #execute interruptible load
        elif order == 7:
            self.intAgent2.getState(self.totalState,self.interruptibleLoadActionMask2)
            self.intAgent2.environment.updateState(self.intAgent2.states,self.interruptibleLoad2)
            self.intAgent2.execute()
            # reward.append(self.intAgent.reward)
            self.updateTotalState("int2")   

    #execute interruptible load
        elif order == 8:
            self.intAgent3.getState(self.totalState,self.interruptibleLoadActionMask3)
            self.intAgent3.environment.updateState(self.intAgent3.states,self.interruptibleLoad3)
            self.intAgent3.execute()
            # reward.append(self.intAgent3.reward)
            self.updateTotalState("int3")   
        
    #execute uninterruptible load
        elif order == 4:
            self.unIntAgent1.getState(self.totalState,self.uninterruptibleLoadActionMask1)
            self.unIntAgent1.environment.updateState(self.unIntAgent1.states,self.uninterruptibleLoad1)
            self.unIntAgent1.execute()
            # reward.append(self.unIntAgent.reward)
            self.updateTotalState("unint1")   

    #execute uninterruptible load
        elif order == 5:
            self.unIntAgent2.getState(self.totalState,self.uninterruptibleLoadActionMask2)
            self.unIntAgent2.environment.updateState(self.unIntAgent2.states,self.uninterruptibleLoad2)
            self.unIntAgent2.execute()
            # reward.append(self.unIntAgent.reward)
            self.updateTotalState("unint2")   

    #execute SOC
        elif order == 0:
            self.socAgent.getState(self.totalState)
            self.socAgent.environment.updateState(self.socAgent.states)
            self.socAgent.execute()
            # reward.append(self.socAgent.reward)
            self.updateTotalState("soc")  
        else:
            self.updateTotalState("none")
        

            
        self.state = self.stateAbstraction(self.totalState)
        done =  bool(sampleTime == 95 and order == 9)

        reward = 0
        states = dict(state=self.state)
        return states, done ,reward

    def stateAbstraction(self, totalState) -> np.array:
        return super().stateAbstraction(totalState)
    
    def updateTotalState(self, mode):
        return super().updateTotalState(mode)