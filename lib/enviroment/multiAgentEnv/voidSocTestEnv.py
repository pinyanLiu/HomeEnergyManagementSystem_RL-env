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
    #interaction
        reward = []
        cost = 0
        soc = soc+delta_soc
        if soc > 1:
            soc = 1
            reward.append(-0.2)
        elif soc < 0 :
            soc = 0
            reward.append(-0.2)
        else:
            #calculate cost proportion   
            if(delta_soc>0):
                cost = pricePerHour * 0.25 * (delta_soc*self.batteryCapacity-pv)
                if cost<0:
                    cost = 0
            elif(delta_soc<=0):
                cost = pricePerHour*0.25*delta_soc*self.batteryCapacity
        if (load+delta_soc*self.batteryCapacity-pv)>self.PgridMax:
            reward.append(-0.2)




        if (sampleTime == 94 and soc <self.socThreshold):
            reward.append(10*(soc-self.socThreshold))

        reward.append(-cost)


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