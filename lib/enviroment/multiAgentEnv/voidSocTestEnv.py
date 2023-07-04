from lib.enviroment.SocTrainEnv import SocEnv
from gym import make
import numpy as np
from random import randint,uniform
 
class VoidSocTest(SocEnv):
    def __init__(self,baseParameter) :
        self.BaseParameter = baseParameter
        self.PgridMax = float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='PgridMax']['value'])[0])
        self.batteryCapacity=float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='batteryCapacity']['value'])[0])
        self.socInit = float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='SOCinit']['value'])[0])
        self.socThreshold = float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='SOCthreshold']['value'])[0])
        
    def states(self):
        return super().states()

    def actions(self):
        return super().actions()
        
    def execute(self, actions):
        sampleTime,load,pv,soc,pricePerHour = self.state
        delta_soc = float(actions)
        reward = []

    #interaction
        cost = 0
        soc = soc+delta_soc
        if soc > 1:
            #delta_soc = 1-soc
            soc = 1
            reward.append(-0.4)
        elif soc < 0 :
            #delta_soc = 0-soc
            soc = 0
            reward.append(-0.4)
        Pgrid = max(0,delta_soc*self.batteryCapacity-pv+load)
        cost = pricePerHour * 0.25 * Pgrid

        if load-pv+delta_soc*self.batteryCapacity>self.PgridMax:
            reward.append(-5)

        if (sampleTime >= 94):
            if(soc < self.socThreshold):
                reward.append(5*(soc-self.socThreshold))
            else:
                reward.append(2)

        reward.append(-0.19*cost+0.1)


        #change to next state
        sampleTime = int(sampleTime+1)
        self.state=np.array([sampleTime,load,pv,soc,pricePerHour])

        #check if all day has done
        self.done = False


        states = dict(state = self.state)

        #REWARD
        self.reward = sum(reward)

        return states,self.done,self.reward        
    
    def reset(self):
        return  np.array([0,0.0,0.0,0.0,0.0])
        

    def updateState(self,states):
        self.state =  states