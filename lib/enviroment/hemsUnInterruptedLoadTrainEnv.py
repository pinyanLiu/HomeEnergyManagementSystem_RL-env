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
        self.info = ImportData(host= self.host ,user= self.user ,passwd= self.passwd ,db= self.db)
        
        #import Base Parameter
        self.BaseParameter = self.info.importBaseParameter()
        self.PgridMax = float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='PgridMax']['value'])[0])


        #import Grid price
        self.allGridPrice = self.info.importGridPrice()
        self.summerGridPrice = self.allGridPrice['summer_price'].tolist()
        self.notSummerGridPrice = self.allGridPrice['not_summer_price'].tolist()

        #pick one day from 360 days
        i = randint(1,359)
        #import Load 
        self.allLoad = self.info.importTrainingLoad()
        self.Load = self.allLoad.iloc[:,i].tolist()
        #import PV
        self.allPV = self.info.importPhotoVoltaic()
        
        if int( i / 30) == 0:
            self.PV = self.allPV['Jan'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif int(i / 30) == 1:
            self.PV = self.allPV['Feb'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif int(i / 30) == 2:
            self.PV = self.allPV['Mar'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif int(i / 30) == 3:
            self.PV = self.allPV['Apr'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif int(i / 30) == 4:
            self.PV = self.allPV['May'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif int(i / 30) == 5:
            self.PV = self.allPV['Jun'].tolist()
            self.GridPrice = self.summerGridPrice
        elif int(i / 30) == 6:
            self.PV = self.allPV['July'].tolist()
            self.GridPrice = self.summerGridPrice
        elif int(i / 30) == 7:
            self.PV = self.allPV['Aug'].tolist()
            self.GridPrice = self.summerGridPrice
        elif int(i / 30) == 8:
            self.PV = self.allPV['Sep'].tolist()
            self.GridPrice = self.summerGridPrice
        elif int(i / 30) == 9:
            self.PV = self.allPV['Oct'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif int(i / 30) == 10:
            self.PV = self.allPV['Nov'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif int(i / 30) == 11:
            self.PV = self.allPV['Dec'].tolist()
            self.GridPrice = self.notSummerGridPrice
        


        self.uninterruptibleLoad = WM(demand=randint(1,5),executePeriod=randint(2,15),AvgPowerConsume=0.3)
        #action Uninterruptible load take (1.on 2.do nothing )
        self.action_space = spaces.Discrete(2)
        #self.observation_space_name = np.array(['sampleTime','load', 'pv', 'pricePerHour' ,'UnInterruptable Remain','switch'])
        #observation space 
        upperLimit = np.array(
            [
                #time block
                95,
                #load
                10.0,
                #PV
                10.0,
                #price per hour
                6.2,
                #Uninterruptible Remain
                75.0,
                #Uninterruptible Switch
                1.0
            ],
            dtype=np.float32,
        )
        lowerLimit = np.array(
            [
                #time block
                0.0,
                #load
                0.0,
                #PV
                0.0,
                #pricePerHour
                0.0,
                #Uninterruptible Remain
                0.0,
                #Uninterruptible Switch
                0.0
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

        #STATE (sampleTime,Load,PV,SOC,pricePerHour,interrupted load remain ,uninterrupted load remain)
        sampleTime,load,pv,pricePerHour,UnRemain,UnSwitch = self.state['states']
        
        # 1.turn on switch 
        if action == 0:
            self.uninterruptibleLoad.turn_on()
            cost = 0.3*(pricePerHour * 0.25 * self.uninterruptibleLoad.AvgPowerConsume*self.uninterruptibleLoad.executePeriod) 
            reward.append(0.1*self.uninterruptibleLoad.executePeriod)

        #2.  do nothing
        elif action == 1 : 
            pass

        assert action == 0 and UnRemain >=0 and UnSwitch != True,'action Invalid'


        # the uninterruptible Load operate itself
        self.uninterruptibleLoad.step()   

        #reward
        reward.append(-0.6*cost)
        if (sampleTime == (94-self.uninterruptibleLoad.executePeriod)) and (self.uninterruptibleLoad.getRemainDemand()!=0):
            reward.append(-0.8*self.uninterruptibleLoad.getRemainDemand())


        #change to next state
        sampleTime = int(sampleTime+1)
        states=np.array([sampleTime,self.Load[sampleTime],self.PV[sampleTime],self.GridPrice[sampleTime],self.uninterruptibleLoad.getRemainDemand(),self.uninterruptibleLoad.switch])
        #action mask
        action_mask = np.asarray([states[4]>0 and states[5]==False,True])
        self.state = dict(states=states,action_mask=action_mask)
        #check if all day is done
        done =  bool(sampleTime == 95)
        #REWARD


        info = {'state':states,'action':action}
        reward = sum(reward)

        return self.state,reward,done,info

        
    def render(self):
        pass
    def reset(self):
        '''
        Starting State
        '''
        self.uninterruptibleLoad = WM(demand=randint(1,5),executePeriod=randint(2,15),AvgPowerConsume=0.3)
        self.GridPrice = (np.random.random(96)*6).tolist()

        #pick one day from 360 days
        i = randint(1,359)
        self.Load = self.allLoad.iloc[:,i].tolist()
        if int( i / 30) == 0:
            self.PV = self.allPV['Jan'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif int(i / 30) == 1:
            self.PV = self.allPV['Feb'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif int(i / 30) == 2:
            self.PV = self.allPV['Mar'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif int(i / 30) == 3:
            self.PV = self.allPV['Apr'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif int(i / 30) == 4:
            self.PV = self.allPV['May'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif int(i / 30) == 5:
            self.PV = self.allPV['Jun'].tolist()
            self.GridPrice = self.summerGridPrice
        elif int(i / 30) == 6:
            self.PV = self.allPV['July'].tolist()
            self.GridPrice = self.summerGridPrice
        elif int(i / 30) == 7:
            self.PV = self.allPV['Aug'].tolist()
            self.GridPrice = self.summerGridPrice
        elif int(i / 30) == 8:
            self.PV = self.allPV['Sep'].tolist()
            self.GridPrice = self.summerGridPrice
        elif int(i / 30) == 9:
            self.PV = self.allPV['Oct'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif int(i / 30) == 10:
            self.PV = self.allPV['Nov'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif int(i / 30) == 11:
            self.PV = self.allPV['Dec'].tolist()
            self.GridPrice = self.notSummerGridPrice


        #reset state
        states=np.array([0,self.Load[0],self.PV[0],self.GridPrice[0],self.uninterruptibleLoad.demand,self.uninterruptibleLoad.switch])
        #action mask
        action_mask = np.asarray([states[4]>0 and states[5]==False,True])
        self.state = dict(states=states,action_mask=action_mask)
        return self.state


if __name__ == '__main__':
    env = make("Hems-v8")
#     # Initialize episode
    states = env.reset()
    done = False
    step = 0
    while not done: # Episode timestep
        actions = env.action_space.sample()
        states, reward, done , info = env.step(action=actions)
        print(info)
        