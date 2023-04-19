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
        sampleTime,soc,remain,pricePreHour,hvacState1,hvacState2,hvacState3,intState,unIntState,intPreference,unintPreference,order = self.state
    #execute HVAC
        if order == 0:
            self.hvacAgent1.getState(self.totalState)
            self.hvacAgent1.environment.updateState(self.hvacAgent1.states)
            self.hvacAgent1.execute()
            # reward.append(self.hvacAgent1.reward)
            self.updateTotalState("hvac1")
        
        elif order ==1:
            self.hvacAgent2.getState(self.totalState)
            self.hvacAgent2.environment.updateState(self.hvacAgent2.states)
            self.hvacAgent2.execute()
            # reward.append(self.hvacAgent2.reward)
            self.updateTotalState("hvac2")

        elif order ==2:
            self.hvacAgent3.getState(self.totalState)
            self.hvacAgent3.environment.updateState(self.hvacAgent3.states)
            self.hvacAgent3.execute()
            # reward.append(self.hvacAgent3.reward)
            self.updateTotalState("hvac3")
        
    #execute interruptible load
        elif order == 3:
            self.intAgent.getState(self.totalState,self.interruptibleLoadActionMask)
            self.intAgent.environment.updateState(self.intAgent.states,self.interruptibleLoad)
            self.intAgent.execute()
            # reward.append(self.intAgent.reward)
            self.updateTotalState("int")   
        
    #execute uninterruptible load
        elif order == 4:
            self.unIntAgent.getState(self.totalState,self.uninterruptibleLoadActionMask)
            self.unIntAgent.environment.updateState(self.unIntAgent.states,self.uninterruptibleLoad)
            self.unIntAgent.execute()
            # reward.append(self.unIntAgent.reward)
            self.updateTotalState("unint")   

    #execute SOC
        elif order == 5:
            self.socAgent.getState(self.totalState)
            self.socAgent.environment.updateState(self.socAgent.states)
            self.socAgent.execute()
            # reward.append(self.socAgent.reward)
            self.updateTotalState("soc")  

        

            
        self.state = self.stateAbstraction(self.totalState)
        done =  bool(sampleTime == 95 and order == 5)

        reward = 0
        states = dict(state=self.state)
        return states, done ,reward

    def stateAbstraction(self, totalState) -> np.array:
        return super().stateAbstraction(totalState)
    
    def updateTotalState(self, mode):
        return super().updateTotalState(mode)