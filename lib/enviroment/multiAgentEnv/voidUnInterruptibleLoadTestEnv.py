from lib.loads.uninterrupted import WM
from gym import make
import numpy as np
from lib.enviroment.UnInterruptibleLoadTrainEnv import UnIntEnv

class VoidUnIntTest(UnIntEnv):
    def __init__(self,baseParameter,unInterruptibleLoad) :
        self.BaseParameter = baseParameter
        self.PgridMax = float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='PgridMax']['value'])[0])
        self.batteryCapacity=float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='batteryCapacity']['value'])[0])
        self.uninterruptibleLoad = unInterruptibleLoad

    def states(self):
        return super().states()

    def actions(self):
        return super().actions()
        
    def execute(self,actions):
        '''
        interaction of each state(changes while taking actions)
        Rewards
        Episode Termination condition
        '''        
        #list for storing reward
        reward = []
        cost = 0
        #STATE (sampleTime,Load,PV,SOC,pricePerHour,Uninterruptible load remain ,uninterruptible load remain)
        sampleTime,load,pv,pricePerHour,deltaSoc,UnRemain,UnSwitch = self.state

        #  do nothing
        if actions == 0:
            pass
        #  turn on switch 
        elif actions == 1 : 
            self.uninterruptibleLoad.turn_on()

        # the uninterruptible Load operate itself
        self.uninterruptibleLoad.step()   
        # if the switch is on , calculate the electricity cost
        if self.uninterruptibleLoad.switch:
            Pess = deltaSoc*self.batteryCapacity
            if Pess<0:
                cost = (pricePerHour * 0.25 * (self.uninterruptibleLoad.AvgPowerConsume-pv+Pess))/self.uninterruptibleLoad.demand
            else:
                cost = (pricePerHour * 0.25 * (self.uninterruptibleLoad.AvgPowerConsume-pv))/self.uninterruptibleLoad.demand
        if cost<0:
            cost = 0 


        #reward
        reward.append(0.07-20*cost)
        if (sampleTime == 94) and (self.uninterruptibleLoad.getRemainDemand()!=0):
            reward.append(-5*self.uninterruptibleLoad.getRemainProcessPercentage())
        

        #change to next state
        sampleTime = int(sampleTime+1)
        self.state=np.array([sampleTime,load,pv,pricePerHour,deltaSoc,self.uninterruptibleLoad.getRemainDemand(),self.uninterruptibleLoad.switch])
        #actions mask
        PgridMaxExceed = (load+deltaSoc+self.uninterruptibleLoad.AvgPowerConsume-pv) >= self.PgridMax

        self.action_mask = np.asarray([True,self.state[5]>0 and self.state[6]==False and not PgridMaxExceed])
        #check if all day is done
        self.done =  bool(sampleTime == 95)
        #REWARD
        self.reward = sum(reward)
        states = dict(state=self.state,action_mask=self.action_mask)
        return states,self.done,self.reward
    

    def reset(self):
        return  np.array([0,0.0,0.0,0.0,0.0,0.0,0.0])
    

    def updateState(self,states,uninterruptibleLoad):
        self.state =  states
        self.uninterruptibleLoad = uninterruptibleLoad