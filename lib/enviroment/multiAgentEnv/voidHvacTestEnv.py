from lib.enviroment.HVACTrainEnv import HvacEnv
import numpy as np

class VoidHvacTest(HvacEnv):
    def __init__(self,baseParameter,allOutdoorTemperature,allUserSetTemperature) :
        self.BaseParameter = baseParameter
        self.outdoorTemperature = allOutdoorTemperature
        self.allUserSetTemperature = allUserSetTemperature
        self.PgridMax = float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='PgridMax']['value'])[0])
        self.epsilon = float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='epsilon']['value'])[0])
        self.eta = float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='eta_HVAC']['value'])[0])
        self.A = float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='A(KW/F)']['value'])[0])
        self.initIndoorTemperature= float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='init_indoor_temperature(F)']['value'])[0])
        self.batteryCapacity=float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='batteryCapacity']['value'])[0])

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
    #STATE (sampleTime,Load,PV,DeltaSOC,pricePerHour,indoor temperature ,outdoor temperature )
        sampleTime,load,pv,pricePerHour,deltaSoc,indoorTemperature,outdoorTemperature,userSetTemperature = self.state
        Power_HVAC = float(actions)
        
    #check if violate pgrid max , if violate, reset the time step until the agent give a action which pass the constrain
        reward = []
        # if load-pv+deltaSoc*self.batteryCapacity+Power_HVAC>self.PgridMax:
        #     reward.append(-5)
        #     states = dict(state=self.state)
        #     self.done = False
        # else:
    #interaction

        #calculate the new indoor temperature for next state
        nextIndoorTemperature = self.epsilon*indoorTemperature+(1-self.epsilon)*(outdoorTemperature-(self.eta/self.A)*Power_HVAC*0.25)

        #calculate proportion
        if (load+Power_HVAC-pv+deltaSoc*self.batteryCapacity) < 0:
            cost = 0
        else:
            cost = Power_HVAC*pricePerHour*0.25

        #temperature reward
        if outdoorTemperature < userSetTemperature :
            r1 = 0
        else :
            r1 = (-pow(indoorTemperature-userSetTemperature,2)+1)/40
        #cost reward
        r2 = -cost/2+0.5

        #REWARD

        reward.append(r1)
        reward.append(r2)
        #Pgrid max reward
        if (load+Power_HVAC-pv+deltaSoc*self.batteryCapacity)>self.PgridMax:
            reward.append(-10)
            
        #change to next state
        sampleTime = int(sampleTime+1)

        #check if all day has done
        self.done = False


        self.state=np.array([sampleTime,load,pv,pricePerHour,deltaSoc,nextIndoorTemperature,outdoorTemperature,userSetTemperature])
        states = dict(state=self.state)


        #set placeholder for infomation
        self.reward = sum(reward)

        return states,self.done,self.reward
    
    def reset(self):
        return  np.array([0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])


    def updateState(self,states):
        self.state =  states










