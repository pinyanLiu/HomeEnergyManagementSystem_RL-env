from lib.enviroment.InterruptibleLoadTrainEnv import IntEnv
from gym import make
import numpy as np
from lib.loads.interrupted import AC

class IntTest(IntEnv):
    def __init__(self) :
        pass

    def states(self):
        return super().states()

    def actions(self):
        return super().actions()
        
    def execute(self,actions,states,interruptibleLoad):
        '''
        interaction of each state(changes while taking action)
        Rewards
        Episode Termination condition
        '''
        #list for storing reward
        reward = []
        cost = 0
        #STATE (sampleTime,Load,PV,DeltaSOC,pricePerHour,interruptible load remain)
        sampleTime,load,pv,pricePerHour,deltaSoc,intRemain = states
        # Turn off switch
        if actions == 0:
            interruptibleLoad.turn_off()
        #  turn on switch 
        elif actions == 1 : 
            interruptibleLoad.turn_on()

        interruptibleLoad.step()

        # if the switch is on , calculate the electricity cost
        if interruptibleLoad.switch:
            Pess = deltaSoc*self.batteryCapacity
            if Pess<0:
                cost = (pricePerHour * 0.25 * (interruptibleLoad.AvgPowerConsume-pv+Pess))/interruptibleLoad.demand
            else:
                cost = (pricePerHour * 0.25 * (interruptibleLoad.AvgPowerConsume-pv))/interruptibleLoad.demand
        if cost<0:
            cost = 0 

        #reward
        reward.append(0.08-10*cost)
        if (sampleTime == 94) and (interruptibleLoad.getRemainDemand()!=0):
            reward.append(-10*interruptibleLoad.getRemainProcessPercentage())


        #change to next state
        sampleTime = int(sampleTime+1)
        self.state=np.array([sampleTime,self.Load[sampleTime],self.PV[sampleTime],self.GridPrice[sampleTime],self.deltaSoc[sampleTime],interruptibleLoad.getRemainDemand()])

        #actions mask
        PgridMaxExceed = (self.Load[sampleTime]+self.deltaSoc[sampleTime]+interruptibleLoad.AvgPowerConsume-self.PV[sampleTime]) >= self.PgridMax
        
        self.action_mask = np.asarray([True,self.state[5]>0 and not PgridMaxExceed])

        #check if all day is done
        self.done =  bool(sampleTime == 95)
        
        #REWARD
        self.reward = sum(reward)
        states = dict(state=self.state,action_mask=self.action_mask)

        return states,self.done,self.reward

        


    def reset(self):
        pass