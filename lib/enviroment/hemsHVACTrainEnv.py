from  gym import spaces
import numpy as np
from random import randint,uniform
from lib.enviroment.hemsTrainEnv import HemsEnv
from gym import make

class HvacEnv(HemsEnv):
    def __init__(self) :
        '''
        Action space
        observation space
        '''
        super().__init__()
        #import Base Parameter
        self.BaseParameter = self.info.importBaseParameter()
        self.PgridMax = float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='PgridMax']['value'])[0])
        self.epsilon = float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='epsilon']['value'])[0])
        self.eta = float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='eta_HVAC']['value'])[0])
        self.A = float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='A(KW/F)']['value'])[0])
        self.max_temperature = float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='max_temperature(F)']['value'])[0])
        self.min_temperature = float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='min_temperature(F)']['value'])[0])
        self.initIndoorTemperature= float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='init_indoor_temperature(F)']['value'])[0])
        self.batteryCapacity=float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='batteryCapacity']['value'])[0])
        self.allUserSetTemperature = self.info.importUserSetTemperatureF()
        self.allOutdoorTemperature = self.info.importTemperatureF()

        self.deltaSoc = [uniform(-0.15,0.15) for _ in range(96)]
        self.GridPrice = [uniform(1.73,6.2) for _ in range(96)]
  
    def states(self):
        upperLimit = np.array(
            [
                #timeblock
                96,
                #load
                10,
                #PV
                10,
                #deltaSoc
                0.15,
                #pricePerHour
                6,
                #indoor temperature
                104,
                #outdoor temperature
                104,
                #user set temperature
                104
            ],
            dtype=np.float32,
        )
        lowerLimit = np.array(
            [
                #timeblock
                0,
                #load
                0,
                #PV
                0,
                #deltaSoc
                -0.15,
                #pricePerHour
                1,
                #indoor temperature
                35,
                #outdoor temperature
                35,
                #user set temperature
                35
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(lowerLimit,upperLimit,dtype=np.float32)
        return dict(type='float',shape=self.observation_space.shape,min_value=lowerLimit,max_value=upperLimit)

    def actions(self):
        #action we take (degree of HVAC power)
        return dict(type='float',shape=(1,),min_value=0,max_value=2)

    def close(self):
        return super().close()
        
    def execute(self,actions):
        '''
        interaction of each state(changes while taking action)
        Rewards
        Episode Termination condition
        '''
    #STATE (sampleTime,Load,PV,DeltaSOC,pricePerHour,indoor temperature ,outdoor temperature )
        sampleTime,load,pv,pricePerHour,deltaSoc,indoorTemperature,outdoorTemperature,userSetTemperature = self.state
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
            r1 = -abs(indoorTemperature-userSetTemperature)/7
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


        self.state=np.array([sampleTime,self.Load[sampleTime],self.PV[sampleTime],self.deltaSoc[sampleTime],self.GridPrice[sampleTime],nextIndoorTemperature,self.outdoorTemperature[sampleTime],self.userSetTemperature[sampleTime]])
        states = dict(state=self.state)


        #set placeholder for infomation
        self.reward = sum(reward)

        return states,self.done,self.reward

        
    def reset(self):
        '''
        Starting State
        '''
        #pick one day from 360 days
        self.i = randint(1,359)
        self.Load = self.allLoad.iloc[:,self.i].tolist()
        #import PV,outTmp,userTmp,GridPrice
        if int( self.i / 30) == 0:
            self.outdoorTemperature = self.allOutdoorTemperature['Jan'].tolist()
            self.userSetTemperature = self.allUserSetTemperature['Jan'].tolist()
            self.PV = self.allPV['Jan'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif int(self.i / 30) == 1:
            self.outdoorTemperature = self.allOutdoorTemperature['Feb'].tolist()
            self.userSetTemperature = self.allUserSetTemperature['Feb'].tolist()
            self.PV = self.allPV['Feb'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif int(self.i / 30) == 2:
            self.outdoorTemperature = self.allOutdoorTemperature['Mar'].tolist()
            self.userSetTemperature = self.allUserSetTemperature['Mar'].tolist()
            self.PV = self.allPV['Mar'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif int(self.i / 30) == 3:
            self.outdoorTemperature = self.allOutdoorTemperature['Apr'].tolist()
            self.PV = self.allPV['Apr'].tolist()
            self.userSetTemperature = self.allUserSetTemperature['Apr'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif int(self.i / 30) == 4:
            self.outdoorTemperature = self.allOutdoorTemperature['May'].tolist()
            self.userSetTemperature = self.allUserSetTemperature['May'].tolist()
            self.PV = self.allPV['May'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif int(self.i / 30) == 5:
            self.outdoorTemperature = self.allOutdoorTemperature['Jun'].tolist()
            self.userSetTemperature = self.allUserSetTemperature['Jun'].tolist()
            self.PV = self.allPV['Jun'].tolist()
            self.GridPrice = self.summerGridPrice
        elif int(self.i / 30) == 6:
            self.outdoorTemperature = self.allOutdoorTemperature['July'].tolist()
            self.userSetTemperature = self.allUserSetTemperature['July'].tolist()
            self.PV = self.allPV['July'].tolist()
            self.GridPrice = self.summerGridPrice
        elif int(self.i / 30) == 7:
            self.outdoorTemperature = self.allOutdoorTemperature['Aug'].tolist()
            self.userSetTemperature = self.allUserSetTemperature['Aug'].tolist()
            self.PV = self.allPV['Aug'].tolist()
            self.GridPrice = self.summerGridPrice
        elif int(self.i / 30) == 8:
            self.outdoorTemperature = self.allOutdoorTemperature['Sep'].tolist()
            self.userSetTemperature = self.allUserSetTemperature['Sep'].tolist()
            self.PV = self.allPV['Sep'].tolist()
            self.GridPrice = self.summerGridPrice
        elif int(self.i / 30) == 9:
            self.outdoorTemperature = self.allOutdoorTemperature['Oct'].tolist()
            self.userSetTemperature = self.allUserSetTemperature['Oct'].tolist()
            self.PV = self.allPV['Oct'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif int(self.i / 30) == 10:
            self.outdoorTemperature = self.allOutdoorTemperature['Nov'].tolist()
            self.userSetTemperature = self.allUserSetTemperature['Nov'].tolist()
            self.PV = self.allPV['Nov'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif int(self.i / 30) == 11:
            self.outdoorTemperature = self.allOutdoorTemperature['Dcb'].tolist()
            self.userSetTemperature = self.allUserSetTemperature['Dcb'].tolist()
            self.PV = self.allPV['Dec'].tolist()
            self.GridPrice = self.notSummerGridPrice

        #reset state
        self.state=np.array([0,self.Load[0],self.PV[0],self.deltaSoc[0],self.GridPrice[0],self.initIndoorTemperature,self.outdoorTemperature[0],self.userSetTemperature[0]])
        return self.state



if __name__ == '__main__':
    env = make("Hems-v6")
#     # Initialize episode
    states = env.reset()
    done = False
    step = 0
    while not done: # Episode timestep
        actions = env.action_space.sample()
        states, reward, done , info = env.step(action=actions)
        print(info)