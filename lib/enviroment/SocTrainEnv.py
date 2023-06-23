from  gym import spaces
import numpy as np
from random import randint,uniform
from lib.enviroment.hemsTrainEnv import HemsEnv
from gym import make

class SocEnv(HemsEnv):
    def __init__(self) :
        '''
        Action space
        observation space
        '''
        super().__init__()

        #import Base Parameter
        self.BaseParameter = self.info.importBaseParameter()
        self.PgridMax = float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='PgridMax']['value'])[0])
        self.batteryCapacity = int(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='batteryCapacity']['value'])[0])
        self.socInit = float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='SOCinit']['value'])[0])
        self.socThreshold = float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='SOCthreshold']['value'])[0])

    def states(self):
        upperLimit = np.array(
            [
                #timeblock
                95,
                #load
                15.0,
                #PV
                10.0,
                #SOC
                1.0,
                #pricePerHour
                6.2
            ],
            dtype=np.float32,
        )
        lowerLimit = np.array(
            [
                #timeblock
                0.0,
                #load
                0.0,
                #PV
                0.0,
                #SOC
                0.0,         
                #pricePerHour
                0.0
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(lowerLimit,upperLimit,dtype=np.float32)
        return dict(type='float',shape=self.observation_space.shape,min_value=lowerLimit,max_value=upperLimit)

    def actions(self):
        #(degree of charging/discharging power)
        return dict(type='float',shape=(1,),min_value=-0.25,max_value=0.25)

    def close(self):
        return super().close()

        
    def execute(self,actions):
        '''
        interaction of each state(changes while taking actions)
        Rewards
        Episode Termination condition
        '''

    #STATE (sampleTime,Load,PV,SOC,pricePerHour,degradationCost)
        sampleTime,load,pv,soc,pricePerHour = self.state
        delta_soc = float(actions)
        # actions(delta_soc) is the degree of charging/discharging power .
        # if delta_soc > 0 means charging , whereas delta_soc < 0 means discharging.
        reward = []

    #interaction
        cost = 0
        soc = soc+delta_soc
        if soc > 1:
            #delta_soc = 1-soc
            soc = 1
            reward.append(-0.2)
        elif soc < 0 :
            #delta_soc = 0-soc
            soc = 0
            reward.append(-0.2)
        Pgrid = max(0,delta_soc*self.batteryCapacity-pv+load)
        cost = pricePerHour * 0.25 * Pgrid

        if load-pv+delta_soc*self.batteryCapacity>self.PgridMax:
            reward.append(-5)



        if (sampleTime == 94):
            if(soc < self.socThreshold):
                reward.append(5*(soc-self.socThreshold))
            else:
                reward.append(2)

        reward.append(-0.16*cost+0.1)


        #change to next state
        sampleTime = int(sampleTime+1)
        self.state=np.array([sampleTime,self.Load[sampleTime],self.PV[sampleTime],soc,self.GridPrice[sampleTime]])

        #check if all day has done
        self.done = bool(sampleTime == 95)


        states = dict(state = self.state)

        #REWARD
        self.reward = sum(reward)

        return states,self.done,self.reward


        


    def reset(self):
        '''
        Starting State
        '''
        #pick one day from 360 days
        self.randomDeltaPrice  = [uniform(-1,1) for _ in range(96)]
        self.randomDeltaPV = [uniform(-0.5,0.5) for _ in range(96)]
        self.i = randint(1,359)
        self.Load = self.allLoad.iloc[:,self.i].tolist()
        if int( self.i / 30) == 0:
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Jan'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
        elif int(self.i / 30) == 1:
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Feb'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
        elif int(self.i / 30) == 2:
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Mar'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
        elif int(self.i / 30) == 3:
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Apr'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
        elif int(self.i / 30) == 4:
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['May'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
        elif int(self.i / 30) == 5:
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Jun'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
        elif int(self.i / 30) == 6:
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['July'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.summerGridPrice,self.randomDeltaPrice) ]
        elif int(self.i / 30) == 7:
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Aug'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.summerGridPrice,self.randomDeltaPrice) ]
        elif int(self.i / 30) == 8:
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Sep'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.summerGridPrice,self.randomDeltaPrice) ]
        elif int(self.i / 30) == 9:
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Oct'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.summerGridPrice,self.randomDeltaPrice) ]
        elif int(self.i / 30) == 10:
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Nov'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
        elif int(self.i / 30) == 11:
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Dec'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]


        #reset state
        self.state=np.array([0,self.Load[0],self.PV[0],self.socInit,self.GridPrice[0]])
        return self.state



if __name__ == '__main__':
    env = make("Hems-v0")
#     # Initialize episode
    states = env.reset()
    done = False
    step = 0
    while not done: # Episode timestep
        actions = env.action_space.sample()
        states, reward, done , info = env.step(actions=actions)
        print(states)
