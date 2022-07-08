from genericpath import samefile
from  gym.envs.Hems.import_data import ImportData 
from gym.envs.Hems.loads.interrupted import AC
from gym.envs.Hems.loads.uninterrupted import WM
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
        self.avgPrice = np.average(self.GridPrice)
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
        self.ac = AC(demand=randint(0,95),AvgPowerConsume=3000)
        wmRandomDemand = randint(5,10)
        self.wm = WM(demand=wmRandomDemand,AvgPowerConsume=3000,executePeriod=wmRandomDemand)
        #action AC and WM take (1.on/on 2. on/off 3.off/on 4.off/off)
        self.action_space = spaces.Discrete(4)
        #self.observation_space_name = np.array(['sampleTime', 'AC','WM','load', 'pv', 'pricePerHour'])
        #observation space 
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
                #AC Remain
                np.finfo(np.float32).max,
                #WM Remain
                np.finfo(np.float32).max
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
                #AC Remain
                np.finfo(np.float32).min,
                #WM Remain
                np.finfo(np.float32).min
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

        #STATE (sampleTime,Load,PV,SOC,pricePerHour,interrupted load remain ,uninterrupted load remain)
        sampleTime,load,pv,pricePerHour,ACRemain,WMRemain = self.state
        

        '''
        There are 3 kind of penalty
            1. Cost . The higher the cost , the higher the penalty
            2. Wrong act in WM . If agent ask WM to stop while WM still haven't reach the Execute period .
            3. Turn on too much . IF agent ask AC or WM to turn on while they have already reach the daily goal

        There is one mix reward (can be reward or penalty , depends on the state and action )
            1. Remain . Get reward if the agent ask AC or WM to turn on while there's still remain time steps need to be turned on . Get penalty if the agent ask AC or WM to turn on while they have already reach the Executed period . 
        '''


        # 1. AC on , WM on
        if action == 0 :
            self.ac.turn_on()
            self.wm.turn_on()
            if ACRemain <= 0 :
                reward.append(-2)
            if WMRemain <= 0:
                reward.append(-0.8*self.wm.executePeriod)
            #avoid WM can't complete execute period if it starts in nearly end of the day
            if (95-sampleTime) < self.wm.executePeriod:
                reward.append(-2)
            reward.append(ACRemain*0.2)
            reward.append(WMRemain*0.2)
            #calculate the cost at this sampletime (multiple 0.25 is for transforming pricePerHour  into per 15 min)
            if (load +self.ac.AvgPowerConsume+self.wm.AvgPowerConsume - pv) < 0:
                cost = -15000 #encourage agent turn on loads when pv is high
            else:
                cost = (pricePerHour-self.avgPrice) * 0.25 *( load +self.ac.AvgPowerConsume+self.wm.AvgPowerConsume - pv  )
            reward.append(-cost/10000)
        
        #2. AC on , WM off
        elif action == 1:
            self.ac.turn_on()
            if ACRemain <= 0:
                reward.append(-2)
            reward.append(ACRemain*0.1)
            if(self.wm.reachExecutePeriod() == False):
                self.wm.turn_on()
                reward.append(WMRemain*0.2)
                reward.append(-2)
                if(load+self.ac.AvgPowerConsume+self.wm.AvgPowerConsume-pv) < 0:
                    cost = -15000 #encourage agent turn on loads when pv is high
                else:
                    cost = (pricePerHour-self.avgPrice) * 0.25 *(load+self.ac.AvgPowerConsume+self.wm.AvgPowerConsume-pv)
            else:
                self.wm.turn_off()
                if (load+self.ac.AvgPowerConsume-pv) < 0:
                    cost = -15000 #encourage agent turn on loads when pv is high
                else:
                    cost = (pricePerHour-self.avgPrice) * 0.25 *(load+self.ac.AvgPowerConsume-pv)
            reward.append(-cost/10000)

        #3. AC off , WM on
        elif action == 2:
            self.ac.turn_off()
            self.wm.turn_on()
            if WMRemain <= 0:
                reward.append(-0.8*self.wm.executePeriod)
            #avoid WM can't complete execute period if it starts in nearly end of the day
            if (95-sampleTime) < self.wm.executePeriod:
                reward.append(-2)

            reward.append(WMRemain*0.2)
            if (load + self.wm.AvgPowerConsume-pv) < 0 :
                cost = -15000 #encourage agent turn on loads when pv is high
            else :
                cost = (pricePerHour-self.avgPrice) * 0.25 * (load + self.wm.AvgPowerConsume-pv)
            reward.append(-cost/10000)

        #4. AC off , WM off
        else : 
            self.ac.turn_off()
            if(self.wm.reachExecutePeriod() == False):
                self.wm.turn_on()
                reward.append(WMRemain*0.2)
                reward.append(-2)
                if (load + self.wm.AvgPowerConsume-pv) < 0:
                    cost = -15000 #encourage agent turn on loads when pv is high
                else :
                    cost = (pricePerHour-self.avgPrice) * 0.25 * (load + self.wm.AvgPowerConsume-pv)
            else:
                if (load-pv)<0:
                    cost = -15000 #encourage agent turn on loads when pv is high
                else:
                    cost = (pricePerHour-self.avgPrice) * 0.25 *(load-pv)
                self.wm.turn_off()
            reward.append(-cost/10000)


        #change to next state
        sampleTime = int(sampleTime+1)
        self.state=np.array([sampleTime,self.Load[sampleTime],self.PV[sampleTime],self.GridPrice[sampleTime],self.ac.getRemainDemand(),self.wm.getRemainDemand()])

        #check if all day is done
        done =  bool(sampleTime == 95)
        #REWARD
        if done == True:
            if self.ac.getRemainDemand() == 0:
                reward.append(40)
            if self.wm.getRemainDemand() == 0:
                reward.append(40)

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
        i = randint(0,359)
        self.Load = self.info.experimentData['Load'].iloc[:,i].tolist()
        self.ac.reset()
        self.wm.reset()
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
        self.state=np.array([0,self.Load[0],self.PV[0],self.GridPrice[0],self.ac.demand,self.wm.demand])
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
        