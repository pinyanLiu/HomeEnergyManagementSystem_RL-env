from  gym.envs.Hems.import_data import ImportData 
from gym.envs.Hems.loads.interrupted import InterruptedLoad,AC
from  gym import Env
from  gym import spaces
from gym import make
import numpy as np
from  yaml import load , SafeLoader
from random import randint

class HemsEnv(Env):
    def __init__(self) :
        '''
        Action space
        observation space
        '''
        #
        # The information of ip should   'NOT'   upload to github
        #
        with open("yaml/mysqlData.yaml","r") as f:
            self.mysqlData = load(f,SafeLoader)

        self.host = self.mysqlData['host']
        self.user = self.mysqlData['user']
        self.passwd = self.mysqlData['passwd']
        self.db = self.mysqlData['db']
        self.info = ImportData(host= self.host ,user= self.user ,passwd= self.passwd ,db= self.db,mode='Training')
        
        self.BaseParameter = self.info.experimentData['BaseParameter']
        self.GridPrice = self.info.experimentData['GridPrice']['price_value'].tolist()
        #pick one day from 360 days
        i = randint(0,359)
        self.Load = self.info.experimentData['Load'].iloc[:,i].tolist()

        if i / 12 == 0:
            self.PV = self.info.experimentData['PV']['Jan'].tolist()
        elif i / 12 == 1:
            self.PV = self.info.experimentData['PV']['Feb'].tolist()
        elif i / 12 == 2:
            self.PV = self.info.experimentData['PV']['Mar'].tolist()
        elif i / 12 == 3:
            self.PV = self.info.experimentData['PV']['Apr'].tolist()
        elif i / 12 == 4:
            self.PV = self.info.experimentData['PV']['May'].tolist()
        elif i / 12 == 5:
            self.PV = self.info.experimentData['PV']['Jun'].tolist()
        elif i / 12 == 6:
            self.PV = self.info.experimentData['PV']['July'].tolist()
        elif i / 12 == 7:
            self.PV = self.info.experimentData['PV']['Aug'].tolist()
        elif i / 12 == 8:
            self.PV = self.info.experimentData['PV']['Sep'].tolist()
        elif i / 12 == 9:
            self.PV = self.info.experimentData['PV']['Oct'].tolist()
        elif i / 12 == 10:
            self.PV = self.info.experimentData['PV']['Nov'].tolist()
        elif i / 12 == 11:
            self.PV = self.info.experimentData['PV']['Dec'].tolist()
        self.ac = AC(demand=8,AvgPowerConsume=3000)
        #action AC take (on,off)
        self.action_space = spaces.Discrete(2)
        #observation space 
        self.observation_space_name = np.array(['sampleTime', 'AC','load', 'pv', 'pricePerHour'])
        upperLimit = np.array(
            [
                #timeblock
                96,
                #load
                np.finfo(np.float32).max,
                #PV
                np.finfo(np.float32).max,
                #pricePerHour
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        lowerLimit = np.array(
            [
                #timeblock
                0,
                #load
                np.finfo(np.float32).min,
                #PV
                np.finfo(np.float32).min,   
                #pricePerHour
                np.finfo(np.float32).min,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(lowerLimit,upperLimit,dtype=np.float32)
        self.state = None
        
    def step(self,action):
        '''
        interaction of each state(changes while taking action)
        Rewards
        Episode Termination condition
        '''
        #error message if getting wrong action
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action),err_msg

        #STATE (sampleTime,Load,PV,SOC,pricePerHour)
        sampleTime,load,pv,pricePerHour = self.state
        
        #interaction
        # if energy supply is greater than consumption
        if pv > load :
            cost = 0.001

        
        # if energy supply is less than consumption
        else:
            # 1. AC on
            if action == 0 :
                self.ac.turn_on()
                #calculate the cost at this sampletime (multiple 0.25 is for transforming pricePerHour  into per 15 min)
                cost = pricePerHour * 0.25 *( load +self.ac.AvgPowerConsume - pv  )
            
            #2. AC off
            else:
                self.ac.turn_off()
                cost = pricePerHour * 0.25 *(load-pv)


        #change to next state
        sampleTime = int(sampleTime+1)
        self.state=np.array([sampleTime,self.Load[sampleTime],self.PV[sampleTime],self.GridPrice[sampleTime]])
        #check if all day is done
        done = bool(
            sampleTime == 95
        )

        #REWARD
        reward = []
        if not done:
            # reward 1
            r1 = -cost/1000
            reward.append(r1)

        # if done
        else : 
            # reward 1
            r1 = -cost/1000
            reward.append(r1)
            #reward 2
            r2= - self.ac.getRemainDemand()
            reward.append(r2)

        reward = sum(reward)



        #set placeholder for infomation
        info = {}

        return self.state,reward,done,info

        
    def render(self):
        pass
    def reset(self):
        '''
        Starting State
        '''
        #pick one day from 360 days
        i = randint(0,359)
        self.Load = self.info.experimentData['Load'].iloc[:,i].tolist()
        self.ac.reset()
        if i % 12 == 0:
            self.PV = self.info.experimentData['PV']['Jan'].tolist()
        elif i % 12 == 1:
            self.PV = self.info.experimentData['PV']['Feb'].tolist()
        elif i % 12 == 2:
            self.PV = self.info.experimentData['PV']['Mar'].tolist()
        elif i % 12 == 3:
            self.PV = self.info.experimentData['PV']['Apr'].tolist()
        elif i % 12 == 4:
            self.PV = self.info.experimentData['PV']['May'].tolist()
        elif i % 12 == 5:
            self.PV = self.info.experimentData['PV']['Jun'].tolist()
        elif i % 12 == 6:
            self.PV = self.info.experimentData['PV']['July'].tolist()
        elif i % 12 == 7:
            self.PV = self.info.experimentData['PV']['Aug'].tolist()
        elif i % 12 == 8:
            self.PV = self.info.experimentData['PV']['Sep'].tolist()
        elif i % 12 == 9:
            self.PV = self.info.experimentData['PV']['Oct'].tolist()
        elif i % 12 == 10:
            self.PV = self.info.experimentData['PV']['Nov'].tolist()
        elif i % 12 == 11:
            self.PV = self.info.experimentData['PV']['Dec'].tolist()


        #reset state
        self.state=np.array([0,self.Load[0],self.PV[0],self.GridPrice[0]])
        return self.state


if __name__ == '__main__':
    env = make("Hems-v0")
#     # Initialize episode
    states = env.reset()
    done = False
    step = 0
    Totalreward = 0
    while not done: # Episode timestep
        actions = env.action_space.sample()
        states, reward, done , info = env.step(action=actions)
        Totalreward += reward
    print(Totalreward)
        