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
        sampleTime,soc,remain,pricePreHour,hvacState,intState,unIntState,intPreference,unintPreference,order = self.state
    #execute HVAC
        if order == 0:
            self.hvacAgent.getState(self.totalState)
            self.hvacAgent.environment.updateState(self.hvacAgent.states)
            self.hvacAgent.execute()
            # reward.append(self.hvacAgent.reward)
            self.updateTotalState("hvac")
        
    #execute interruptible load
        elif order == 2:
            self.intAgent.getState(self.totalState,self.interruptibleLoadActionMask)
            self.intAgent.environment.updateState(self.intAgent.states,self.interruptibleLoad)
            self.intAgent.execute()
            # reward.append(self.intAgent.reward)
            self.updateTotalState("int")   
        
    #execute uninterruptible load
        elif order == 1:
            self.unIntAgent.getState(self.totalState,self.uninterruptibleLoadActionMask)
            self.unIntAgent.environment.updateState(self.unIntAgent.states,self.uninterruptibleLoad)
            self.unIntAgent.execute()
            # reward.append(self.unIntAgent.reward)
            self.updateTotalState("unint")   

    #execute SOC
        elif order == 3:
            self.socAgent.getState(self.totalState)
            self.socAgent.environment.updateState(self.socAgent.states)
            self.socAgent.execute()
            # reward.append(self.socAgent.reward)
            self.updateTotalState("soc")  

            
        self.state = self.stateAbstraction(self.totalState)
        done =  bool(sampleTime == 95 and order == 3)

        reward = 0
        states = dict(state=self.state)
        return states, done ,reward

    def stateAbstraction(self, totalState) -> np.array:
        return super().stateAbstraction(totalState)
    
    def updateTotalState(self, mode):
        return super().updateTotalState(mode)