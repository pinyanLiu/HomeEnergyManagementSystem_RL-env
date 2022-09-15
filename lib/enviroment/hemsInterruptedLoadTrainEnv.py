from  gym.envs.Hems.import_data import ImportData 
from gym.envs.Hems.loads.interrupted import AC
from  gym import Env,spaces,make
import numpy as np
from  yaml import load , SafeLoader
from random import randint,uniform

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
        self.info = ImportData(host= self.host ,user= self.user ,passwd= self.passwd ,db= self.db)
        
        #import Base Parameter
        self.BaseParameter = self.info.importBaseParameter()
        self.PgridMax = float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='PgridMax']['value'])[0])
        self.batteryCapacity = int(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='batteryCapacity']['value'])[0])


        #import Grid price
        self.GridPrice = self.info.importGridPrice()
        self.GridPrice = self.GridPrice['price_value'].tolist()
        
        #pick one day from 360 days
        i = randint(1,359)
        #import Load 
        self.allLoad = self.info.importTrainingLoad()
        self.Load = self.allLoad.iloc[:,i].tolist()
        self.allPV = self.info.importPhotoVoltaic()
        #import PV
        if int( i / 30) == 0:
            self.PV = self.allPV['Jan'].tolist()
        elif int(i / 30) == 1:
            self.PV = self.allPV['Feb'].tolist()
        elif int(i / 30) == 2:
            self.PV = self.allPV['Mar'].tolist()
        elif int(i / 30) == 3:
            self.PV = self.allPV['Apr'].tolist()
        elif int(i / 30) == 4:
            self.PV = self.allPV['May'].tolist()
        elif int(i / 30) == 5:
            self.PV = self.allPV['Jun'].tolist()
        elif int(i / 30) == 6:
            self.PV = self.allPV['July'].tolist()
        elif int(i / 30) == 7:
            self.PV = self.allPV['Aug'].tolist()
        elif int(i / 30) == 8:
            self.PV = self.allPV['Sep'].tolist()
        elif int(i / 30) == 9:
            self.PV = self.allPV['Oct'].tolist()
        elif int(i / 30) == 10:
            self.PV = self.allPV['Nov'].tolist()
        elif int(i / 30) == 11:
            self.PV = self.allPV['Dec'].tolist()
        self.interruptableLoad = AC(demand=randint(1,40),AvgPowerConsume=uniform(0.5,1.5))
        # Interruptable load's actions  ( 1.on 2.off )
        self.action_space = spaces.Discrete(2)
        #self.observation_space_name = np.array(['sampleTime','load', 'pv', 'pricePerHour' ,'Interruptable Remain'])
        #observation space 
        upperLimit = np.array(
            [
                #timeblock
                95,
                #load
                10.0,
                #PV
                10.0,
                #pricePerHour
                6.0,
                #Interruptable Remain
                40.0,
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
                #pricePerHour
                1.0,
                #Interruptable Remain
                -56.0,
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

        #list for storing reward
        reward = []
        cost = 0

        #STATE (sampleTime,Load,PV,SOC,pricePerHour,interrupted load remain)
        sampleTime,load,pv,pricePerHour,IntRemain, = self.state
        

        '''
        There are 3 kind of penalty
            1. Cost . The higher the cost , the higher the penalty
            2. Wrong act in WM . If agent ask WM to stop while WM still haven't reach the Execute period .
            3. Turn on too much . IF agent ask AC or WM to turn on while they have already reach the daily goal

        There is one mix reward (can be reward or penalty , depends on the state and action )
            1. Remain . Get reward if the agent ask AC or WM to turn on while there's still remain time steps need to be turned on . Get penalty if the agent ask AC or WM to turn on while they have already reach the Executed period . 
        '''


        # 1. on 
        if action == 0 :
            self.interruptableLoad.turn_on()
            #calculate the cost at this sampletime (multiple 0.25 is for transforming pricePerHour  into per 15 min)
            if (load +self.interruptableLoad.AvgPowerConsume - pv) < 0:
                cost = 0 #encourage agent turn on loads when pv is high
            #PgridMax reward
            elif(load+self.interruptableLoad.AvgPowerConsume-pv>self.PgridMax):
                reward.append(-1)

            #calculate cost and proportion
            else:
                proportion = np.abs(self.interruptableLoad.AvgPowerConsume / (load + self.interruptableLoad.AvgPowerConsume - pv) )
                cost = proportion*(pricePerHour * 0.25 *( load + self.interruptableLoad.AvgPowerConsume - pv ))  

        #2.  off
        elif action == 1 : 
            self.interruptableLoad.turn_off()
            cost = 0

        reward.append(-cost)

        #change to next state
        sampleTime = int(sampleTime+1)
        self.state=np.array([sampleTime,self.Load[sampleTime],self.PV[sampleTime],self.GridPrice[sampleTime],self.interruptableLoad.getRemainDemand()])

        #check if all day is done
        done =  bool(sampleTime == 95)
        #REWARD
        if done == True:
            if IntRemain == 0:
                reward.append(10)
            else:
                reward.append(-10)


        reward = sum(reward)
        info = {'reward':reward}

        return self.state,reward,done,info

        
    def render(self):
        pass
    def reset(self):
        '''
        Starting State
        '''
        #pick one day from 360 days
        i = randint(1,359)
        self.Load = self.allLoad.iloc[:,i].tolist()
        self.interruptableLoad.reset()
        self.interruptableLoad = AC(demand=randint(1,40),AvgPowerConsume=uniform(0.5,1.5))
        if int( i / 30) == 0:
            self.PV = self.allPV['Jan'].tolist()
        elif int(i / 30) == 1:
            self.PV = self.allPV['Feb'].tolist()
        elif int(i / 30) == 2:
            self.PV = self.allPV['Mar'].tolist()
        elif int(i / 30) == 3:
            self.PV = self.allPV['Apr'].tolist()
        elif int(i / 30) == 4:
            self.PV = self.allPV['May'].tolist()
        elif int(i / 30) == 5:
            self.PV = self.allPV['Jun'].tolist()
        elif int(i / 30) == 6:
            self.PV = self.allPV['July'].tolist()
        elif int(i / 30) == 7:
            self.PV = self.allPV['Aug'].tolist()
        elif int(i / 30) == 8:
            self.PV = self.allPV['Sep'].tolist()
        elif int(i / 30) == 9:
            self.PV = self.allPV['Oct'].tolist()
        elif int(i / 30) == 10:
            self.PV = self.allPV['Nov'].tolist()
        elif int(i / 30) == 11:
            self.PV = self.allPV['Dec'].tolist()

        #reset state
        self.state=np.array([0,self.Load[0],self.PV[0],self.GridPrice[0],self.interruptableLoad.demand])
        return self.state


if __name__ == '__main__':
    env = make("Hems-v4")
#     # Initialize episode
    states = env.reset()
    done = False
    step = 0
    while not done: # Episode timestep
        actions = env.action_space.sample()
        states, reward, done , info = env.step(action=actions)
        print(info)
        