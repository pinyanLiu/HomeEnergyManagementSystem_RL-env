from lib.enviroment.InterruptibleLoadTrainEnv import IntEnv
from gym import make
import numpy as np
from lib.loads.interrupted import AC

class VoidIntTest(IntEnv):
    def __init__(self,baseParameter,interruptibleLoad) :
        self.BaseParameter = baseParameter
        self.PgridMax = float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='PgridMax']['value'])[0])
        self.batteryCapacity=float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='batteryCapacity']['value'])[0])
        self.interruptibleLoad = interruptibleLoad

    def states(self):
        return super().states()

    def actions(self):
        return super().actions()
        
    def execute(self,actions):
        '''
        interaction of each state(changes while taking action)
        Rewards
        Episode Termination condition
        '''
        #list for storing reward
        reward = []
        cost = 0
        #STATE (sampleTime,Load,PV,DeltaSOC,pricePerHour,interruptible load remain)
        sampleTime,load,pv,pricePerHour,deltaSoc,intRemain,intUserPreference = self.state["state"]
        # Turn off switch
        if actions == 0:
            self.interruptibleLoad.turn_off()
        #  turn on switch 
        elif actions == 1 : 
            self.interruptibleLoad.turn_on()

        self.interruptibleLoad.step()

        # if the switch is on , calculate the electricity cost
        if self.interruptibleLoad.switch:
            Pess = deltaSoc*self.batteryCapacity
            if Pess<0:
                cost = (pricePerHour * 0.25 * (self.interruptibleLoad.AvgPowerConsume-pv+Pess))/self.interruptibleLoad.demand
            else:
                cost = (pricePerHour * 0.25 * (self.interruptibleLoad.AvgPowerConsume-pv))/self.interruptibleLoad.demand
        if cost<0:
            cost = 0 

        #reward
        reward.append(0.08-10*cost)
        if (sampleTime == 94) and (self.interruptibleLoad.getRemainDemand()!=0):
            reward.append(-10*self.interruptibleLoad.getRemainProcessPercentage())


        #change to next state
        sampleTime = int(sampleTime+1)
        self.state=np.array([sampleTime,load,pv,pricePerHour,deltaSoc,self.interruptibleLoad.getRemainDemand(),intUserPreference])

        #actions mask
        PgridMaxExceed = (load+deltaSoc+self.interruptibleLoad.AvgPowerConsume-pv) >= self.PgridMax
        
        self.action_mask = np.asarray([True,self.interruptibleLoad.getRemainDemand()>0 and not PgridMaxExceed])

        #check if all day is done
        self.done =  False
        
        #REWARD
        self.reward = sum(reward)
        states = dict(state=self.state,action_mask=self.action_mask)

        return states,self.done,self.reward

        


    def reset(self):
        return  np.array([0,0.0,0.0,0.0,0.0,0.0,0.0])
    

    def updateState(self,states,interruptibleLoad):
        self.state =  states
        self.interruptibleLoad = interruptibleLoad