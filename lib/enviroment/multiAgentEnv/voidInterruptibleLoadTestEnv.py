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
        
    def execute(self, actions,states):

        reward = []
        cost = 0
        sampleTime,load,pv,pricePerHour,deltaSoc,intRemain = states
        # Turn off switch
        if actions == 0:
            self.interruptibleLoad.turn_off()
        #  turn on switch 
        elif actions == 1 : 
            self.interruptibleLoad.turn_on()
        self.interruptibleLoad.step()
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

        


    def reset(self):
        pass