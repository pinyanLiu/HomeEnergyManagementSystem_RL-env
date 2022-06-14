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
        self.ac = AC(demand=5,AvgPowerConsume=3000)
        self.wm = WM(demand=6,AvgPowerConsume=3000,executePeriod=6)
        #action AC take (on,off)
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
        fail = False
        

        # 1. AC on , WM on
        if action == 0 :
            self.ac.turn_on()
            self.wm.turn_on()
            reward.append(0.5)
            #calculate the cost at this sampletime (multiple 0.25 is for transforming pricePerHour  into per 15 min)
            cost = pricePerHour * 0.25 *( load +self.ac.AvgPowerConsume+self.wm.AvgPowerConsume - pv  )
        
        #2. AC on , WM off
        elif action == 1:
            self.ac.turn_on()
            if(self.wm.reachExecutePeriod() == False):
                reward.append(-0.5)
            else:
                reward.append(0.5)
            self.wm.turn_off()
            cost = pricePerHour * 0.25 *(load+self.ac.AvgPowerConsume-pv)

        #3. AC off , WM on
        elif action == 2:
            self.ac.turn_off()
            self.wm.turn_on()
            fail=False
            reward.append(0.5)
            cost = pricePerHour * 0.25 * (load + self.wm.AvgPowerConsume-pv)

        #4. AC off , WM off
        else : 
            self.ac.turn_off()
            if(self.wm.reachExecutePeriod() == False):
                #fail = True
                reward.append(-0.5)

            else:
                reward.append(0.5)
            self.wm.turn_off()
            cost = pricePerHour * 0.25 *(load-pv)

        if WMRemain<0:
            reward.append(-np.abs(WMRemain*0.1))
        if ACRemain<0:
            reward.append(-np.abs(ACRemain*0.1))


        #change to next state
        sampleTime = int(sampleTime+1)
        self.state=np.array([sampleTime,self.Load[sampleTime],self.PV[sampleTime],self.GridPrice[sampleTime],self.ac.getRemainDemand(),self.wm.getRemainDemand()])

        #check if all day is done
        done =  bool(sampleTime == 95)
        #REWARD
        if done == True:
            if self.ac.getRemainDemand()==0:
                reward.append(20)
            if self.wm.getRemainDemand()==0:
                reward.append(20)

        reward = sum(reward)

        info = {ACRemain,WMRemain}
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
    Totalreward = 0
    while not done: # Episode timestep
        actions = env.action_space.sample()
        states, reward, done , info = env.step(action=actions)
        print(info)
        