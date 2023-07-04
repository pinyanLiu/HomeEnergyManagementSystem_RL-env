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

        if actions == 0 :
            delta_soc = 0.25
        elif actions ==1:
            delta_soc = 0.2
        elif actions ==2:
            delta_soc = 0.15
        elif actions ==3:
            delta_soc = 0.1
        elif actions ==4:
            delta_soc = 0.05
        elif actions ==5:
            delta_soc = 0.00
        elif actions ==6:
            delta_soc = -0.05
        elif actions ==7:
            delta_soc = -0.1
        elif actions ==8:
            delta_soc = -0.15
        elif actions ==9:
            delta_soc = -0.2
        elif actions ==10:
            delta_soc = -0.25

        # actions(delta_soc) is the degree of charging/discharging power .
        # if delta_soc > 0 means charging , whereas delta_soc < 0 means discharging.
        reward = []

    #interaction
        cost = 0
        soc = soc+delta_soc

        Pgrid = max(0,delta_soc*self.batteryCapacity-pv+load)
        cost = pricePerHour * 0.25 * Pgrid

        if load-pv+delta_soc*self.batteryCapacity>self.PgridMax:
            reward.append(-5)

        socMask = [1-soc>0.25,1-soc>0.2,1-soc>0.15,1-soc>0.1,1-soc>0.05,True,soc>0.05,soc>0.1,soc>0.15,soc>0.2,soc>0.25]
        self.action_mask = np.asarray(socMask)

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


        states = dict(state = self.state,action_mask = self.action_mask)

        #REWARD
        self.reward = sum(reward)

        return states,self.done,self.reward        
    
    def reset(self):
        return  np.array([0,0.0,0.0,0.0,0.0])
        

    def updateState(self,states):
        self.state =  states