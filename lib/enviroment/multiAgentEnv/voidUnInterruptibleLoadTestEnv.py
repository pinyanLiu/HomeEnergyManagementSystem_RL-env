from lib.loads.uninterrupted import WM
from gym import make
import numpy as np
from lib.enviroment.UnInterruptibleLoadTrainEnv import UnIntEnv

class UnIntTest(UnIntEnv):
    def __init__(self) :
        pass

    def states(self):
        return super().states()

    def actions(self):
        return super().actions()
        
    def execute(self,actions,states,uninterruptibleLoad):
        '''
        interaction of each state(changes while taking actions)
        Rewards
        Episode Termination condition
        '''        
        #list for storing reward
        reward = []
        cost = 0
        #STATE (sampleTime,Load,PV,SOC,pricePerHour,Uninterruptible load remain ,uninterruptible load remain)
        sampleTime,load,pv,pricePerHour,deltaSoc,UnRemain,UnSwitch = states

        #  do nothing
        if actions == 0:
            pass
        #  turn on switch 
        elif actions == 1 : 
            uninterruptibleLoad.turn_on()

        # the uninterruptible Load operate itself
        uninterruptibleLoad.step()   
        # if the switch is on , calculate the electricity cost
        if uninterruptibleLoad.switch:
            Pess = deltaSoc*self.batteryCapacity
            if Pess<0:
                cost = (pricePerHour * 0.25 * (uninterruptibleLoad.AvgPowerConsume-pv+Pess))/uninterruptibleLoad.demand
            else:
                cost = (pricePerHour * 0.25 * (uninterruptibleLoad.AvgPowerConsume-pv))/uninterruptibleLoad.demand
        if cost<0:
            cost = 0 


        #reward
        reward.append(0.07-20*cost)
        if (sampleTime == 94) and (uninterruptibleLoad.getRemainDemand()!=0):
            reward.append(-5*uninterruptibleLoad.getRemainProcessPercentage())
        

        #change to next state
        sampleTime = int(sampleTime+1)
        self.state=np.array([sampleTime,self.Load[sampleTime],self.PV[sampleTime],self.GridPrice[sampleTime],self.deltaSoc[sampleTime],uninterruptibleLoad.getRemainDemand(),uninterruptibleLoad.switch])
        #actions mask
        PgridMaxExceed = (self.Load[sampleTime]+self.deltaSoc[sampleTime]+uninterruptibleLoad.AvgPowerConsume-self.PV[sampleTime]) >= self.PgridMax

        self.action_mask = np.asarray([True,self.state[5]>0 and self.state[6]==False and not PgridMaxExceed])
        #check if all day is done
        self.done =  bool(sampleTime == 95)
        #REWARD
        self.reward = sum(reward)
        states = dict(state=self.state,action_mask=self.action_mask)
        return states,self.done,self.reward
    
    def reset(self):
        pass