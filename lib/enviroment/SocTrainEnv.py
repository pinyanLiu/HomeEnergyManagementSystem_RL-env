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

        self.GridPrice = [uniform(1.73,6.2) for _ in range(96)]

    def states(self):
        upperLimit = np.array(
            [
                #timeblock
                95,
                #load
                10.0,
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
        return dict(type='float',shape=(1,),min_value=-0.15,max_value=0.15)

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


    #interaction
        reward = []
        cost = 0
        soc = soc+delta_soc
        if soc > 1:
            soc = 1
            reward.append(-0.5)
        elif soc < 0 :
            soc = 0
            reward.append(-0.5)
        else:
            #calculate cost proportion   
            if(delta_soc>0):
                cost = pricePerHour * 0.25 * (delta_soc*self.batteryCapacity-pv)
                if cost<0:
                    cost = 0
            elif(delta_soc<=0):
                cost = pricePerHour*0.25*delta_soc*self.batteryCapacity
        if (load+delta_soc*self.batteryCapacity-pv)>self.PgridMax:
            reward.append(-0.5)




        if (sampleTime == 94 and soc <self.socThreshold):
            reward.append(10*(soc-self.socThreshold))

        reward.append(-cost)


        #change to next state
        sampleTime = int(sampleTime+1)
        self.state=np.array([sampleTime,self.Load[sampleTime],self.PV[sampleTime],soc,self.GridPrice[sampleTime]])

        #check if all day has done
        self.done = bool(sampleTime == 95)


        states = dict(state = self.state)
#print(reward)

        #REWARD
        self.reward = sum(reward)

        return states,self.done,self.reward

        


    def reset(self):
        '''
        Starting State
        '''
        #pick one day from 360 days
        self.i = randint(1,359)
        self.Load = self.allLoad.iloc[:,self.i].tolist()
        if int( self.i / 30) == 0:
            self.PV = self.allPV['Jan'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif int(self.i / 30) == 1:
            self.PV = self.allPV['Feb'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif int(self.i / 30) == 2:
            self.PV = self.allPV['Mar'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif int(self.i / 30) == 3:
            self.PV = self.allPV['Apr'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif int(self.i / 30) == 4:
            self.PV = self.allPV['May'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif int(self.i / 30) == 5:
            self.PV = self.allPV['Jun'].tolist()
            self.GridPrice = self.summerGridPrice
        elif int(self.i / 30) == 6:
            self.PV = self.allPV['July'].tolist()
            self.GridPrice = self.summerGridPrice
        elif int(self.i / 30) == 7:
            self.PV = self.allPV['Aug'].tolist()
            self.GridPrice = self.summerGridPrice
        elif int(self.i / 30) == 8:
            self.PV = self.allPV['Sep'].tolist()
            self.GridPrice = self.summerGridPrice
        elif int(self.i / 30) == 9:
            self.PV = self.allPV['Oct'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif int(self.i / 30) == 10:
            self.PV = self.allPV['Nov'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif int(self.i / 30) == 11:
            self.PV = self.allPV['Dec'].tolist()
            self.GridPrice = self.notSummerGridPrice


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
