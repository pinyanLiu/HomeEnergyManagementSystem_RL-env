from lib.enviroment.HVACTrainEnv import HvacEnv
import numpy as np

class VoidHvacTest(HvacEnv):
    def __init__(self) :
        pass

    def states(self):
        return super().states()

    def actions(self):
        return super().actions()
        
    def execute(self,actions,states):
        '''
        interaction of each state(changes while taking action)
        Rewards
        Episode Termination condition
        '''
    #STATE (sampleTime,Load,PV,DeltaSOC,pricePerHour,indoor temperature ,outdoor temperature )
        sampleTime,load,pv,pricePerHour,deltaSoc,indoorTemperature,outdoorTemperature,userSetTemperature = states
        Power_HVAC = float(actions)


    #interaction

        #calculate the new indoor temperature for next state
        nextIndoorTemperature = self.epsilon*indoorTemperature+(1-self.epsilon)*(outdoorTemperature-(self.eta/self.A)*Power_HVAC)

        #calculate proportion
        if (load+Power_HVAC-pv+deltaSoc*self.batteryCapacity) < 0:
            cost = 0
        else:
            cost = Power_HVAC*pricePerHour*0.25

        #temperature reward
        if indoorTemperature > userSetTemperature :
            r1 = -pow(indoorTemperature-userSetTemperature,2)/5
        else :
            if indoorTemperature >= outdoorTemperature:
                r1 = 0.01
            r1 = 0
        #cost reward
        r2 = -cost/2


        #REWARD
        reward = []
        reward.append(r1)
        reward.append(r2)
        #Pgrid max reward
        if (load+Power_HVAC-pv+deltaSoc*self.batteryCapacity)>self.PgridMax:
            reward.append(-1)
            
        #change to next state
        sampleTime = int(sampleTime+1)

        #check if all day has done
        self.done = bool(
            sampleTime == 95
        )


        self.state=np.array([sampleTime,self.Load[sampleTime],self.PV[sampleTime],self.GridPrice[sampleTime],self.deltaSoc[sampleTime],nextIndoorTemperature,self.outdoorTemperature[sampleTime],self.userSetTemperature[sampleTime]])
        states = dict(state=self.state)


        #set placeholder for infomation
        self.reward = sum(reward)

        return states,self.done,self.reward
    
    def reset(self):
        pass












