from  gym import spaces
import numpy as np
from random import randint,uniform
from lib.enviroment.hemsTrainEnv import HemsEnv

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
        self.initIndoorTemperature= float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='init_indoor_temperature(F)']['value'])[0])
        self.batteryCapacity=float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='batteryCapacity']['value'])[0])
        self.allUserSetTemperature = self.info.importUserSetTemperatureF()
        self.allOutdoorTemperature = self.info.importTemperatureF()
        self.deltaSoc = self.allDeltaSOC['Jan']

  
    def states(self):
        upperLimit = np.array(
            [
                #timeblock
                95,
                #load
                10,
                #PV
                10,
                #pricePerHour
                6.2,
                #deltaSoc
                0.15,
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
                #pricePerHour
                1,
                #deltaSoc
                -0.15,
                #indoor temperature
                20,
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
        '''
        Starting State
        '''
        #pick one day from 360 days
        self.randomTemperature = [uniform(-2,2)for _ in range(96)]
        self.randomDeltaPrice  = [uniform(-0.25,0.25) for _ in range(96)]
        self.randomDeltaPV = [uniform(-0.5,0.5) for _ in range(96)]
        self.randomDeltaSOC = [uniform(-0.05,0.05) for _ in range(96)]
        self.i = randint(1,359)
        self.Load = self.allLoad.iloc[:,self.i].tolist()
        #import PV,outTmp,userTmp,GridPrice
        if int( self.i / 30) == 0:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['Jan'].tolist(),self.randomTemperature)]
            self.userSetTemperature = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature['Jan'].tolist(),self.randomTemperature)]
            self.deltaSOC = [min(max(x+y,-2.5),2.5) for x,y in zip(self.allDeltaSOC['Jan'].tolist(),self.randomDeltaSOC)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Jan'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
        elif int(self.i / 30) == 1:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['Feb'].tolist(),self.randomTemperature)]
            self.userSetTemperature = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature['Feb'].tolist(),self.randomTemperature)]
            self.deltaSOC = [min(max(x+y,-2.5),2.5) for x,y in zip(self.allDeltaSOC['Feb'].tolist(),self.randomDeltaSOC)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Feb'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
        elif int(self.i / 30) == 2:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['Mar'].tolist(),self.randomTemperature)]
            self.userSetTemperature = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature['Mar'].tolist(),self.randomTemperature)]
            self.deltaSOC = [min(max(x+y,-2.5),2.5) for x,y in zip(self.allDeltaSOC['Mar'].tolist(),self.randomDeltaSOC)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Mar'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
        elif int(self.i / 30) == 3:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['Apr'].tolist(),self.randomTemperature)]
            self.userSetTemperature = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature['Apr'].tolist(),self.randomTemperature)]
            self.deltaSOC = [min(max(x+y,-2.5),2.5) for x,y in zip(self.allDeltaSOC['Apr'].tolist(),self.randomDeltaSOC)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Apr'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
        elif int(self.i / 30) == 4:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['May'].tolist(),self.randomTemperature)]
            self.userSetTemperature = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature['May'].tolist(),self.randomTemperature)]
            self.deltaSOC = [min(max(x+y,-2.5),2.5) for x,y in zip(self.allDeltaSOC['May'].tolist(),self.randomDeltaSOC)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['May'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
        elif int(self.i / 30) == 5:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['Jun'].tolist(),self.randomTemperature)]
            self.userSetTemperature = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature['Jun'].tolist(),self.randomTemperature)]
            self.deltaSOC = [min(max(x+y,-2.5),2.5) for x,y in zip(self.allDeltaSOC['Jun'].tolist(),self.randomDeltaSOC)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Jun'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
        elif int(self.i / 30) == 6:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['July'].tolist(),self.randomTemperature)]
            self.userSetTemperature = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature['July'].tolist(),self.randomTemperature)]
            self.deltaSOC = [min(max(x+y,-2.5),2.5) for x,y in zip(self.allDeltaSOC['July'].tolist(),self.randomDeltaSOC)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['July'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.summerGridPrice,self.randomDeltaPrice) ]
        elif int(self.i / 30) == 7:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['Aug'].tolist(),self.randomTemperature)]
            self.userSetTemperature = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature['Aug'].tolist(),self.randomTemperature)]
            self.deltaSOC = [min(max(x+y,-2.5),2.5) for x,y in zip(self.allDeltaSOC['Aug'].tolist(),self.randomDeltaSOC)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Aug'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.summerGridPrice,self.randomDeltaPrice) ]
        elif int(self.i / 30) == 8:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['Sep'].tolist(),self.randomTemperature)]
            self.userSetTemperature = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature['Sep'].tolist(),self.randomTemperature)]
            self.deltaSOC = [min(max(x+y,-2.5),2.5) for x,y in zip(self.allDeltaSOC['Sep'].tolist(),self.randomDeltaSOC)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Sep'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.summerGridPrice,self.randomDeltaPrice) ]
        elif int(self.i / 30) == 9:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['Oct'].tolist(),self.randomTemperature)]
            self.userSetTemperature = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature['Oct'].tolist(),self.randomTemperature)]
            self.deltaSOC = [min(max(x+y,-2.5),2.5) for x,y in zip(self.allDeltaSOC['Oct'].tolist(),self.randomDeltaSOC)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Oct'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.summerGridPrice,self.randomDeltaPrice) ]
        elif int(self.i / 30) == 10:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['Nov'].tolist(),self.randomTemperature)]
            self.userSetTemperature = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature['Nov'].tolist(),self.randomTemperature)]
            self.deltaSOC = [min(max(x+y,-2.5),2.5) for x,y in zip(self.allDeltaSOC['Nov'].tolist(),self.randomDeltaSOC)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Nov'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
        elif int(self.i / 30) == 11:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['Dcb'].tolist(),self.randomTemperature)]
            self.userSetTemperature = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature['Dcb'].tolist(),self.randomTemperature)]
            self.deltaSOC = [min(max(x+y,-2.5),2.5) for x,y in zip(self.allDeltaSOC['Dcb'].tolist(),self.randomDeltaSOC)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Dec'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]

        #reset state
        self.state=np.array([0,self.Load[0],self.PV[0],self.GridPrice[0],self.deltaSoc[0],self.initIndoorTemperature,self.outdoorTemperature[0],self.userSetTemperature[0]])
        return self.state


