from  gym.envs.Hems.import_data import ImportData 
from gym.envs.Hems.loads.interrupted import AC
from gym.envs.Hems.loads.uninterrupted import WM
from  gym import Env
from  gym import spaces
from gym import make
import numpy as np
from  yaml import load , SafeLoader

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
        self.GridPrice = self.info.importGridPrice()
        self.GridPrice = self.GridPrice['price_value'].tolist()

        #each month pick one day for testing
        self.i = 0
        #import Load 
        self.allLoad = self.info.importTestingLoad()
        self.Load = self.allLoad.iloc[:,self.i].tolist()

        #import PV
        self.allPV = self.info.importPhotoVoltaic()        
        if self.i  == 0:
            self.PV = self.allPV['Jan'].tolist()
        elif self.i  == 1:
            self.PV = self.allPV['Feb'].tolist()
        elif self.i  == 2:
            self.PV = self.allPV['Mar'].tolist()
        elif self.i  == 3:
            self.PV = self.allPV['Apr'].tolist()
        elif self.i  == 4:
            self.PV = self.allPV['May'].tolist()
        elif self.i  == 5:
            self.PV = self.allPV['Jun'].tolist()
        elif self.i  == 6:
            self.PV = self.allPV['July'].tolist()
        elif self.i  == 7:
            self.PV = self.allPV['Aug'].tolist()
        elif self.i  == 8:
            self.PV = self.allPV['Sep'].tolist()
        elif self.i  == 9:
            self.PV = self.allPV['Oct'].tolist()
        elif self.i  == 10:
            self.PV = self.allPV['Nov'].tolist()
        elif self.i  == 11:
            self.PV = self.allPV['Dec'].tolist()

        self.uninterruptibleLoad = WM(demand=3,executePeriod=5,AvgPowerConsume=1.5)


        #action Uninterruptible load take (1.on 2.do nothing )
        self.action_space = spaces.Discrete(2)
        #self.observation_space_name = np.array(['sampleTime','load', 'pv', 'pricePerHour' ,'UnInterruptable Remain'])
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
                50.0,
                #Interruptable Switch
                1.0
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
                0.0,
                #Interruptable Remain
                0.0,
                #Interruptable Switch
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

        #STATE (sampleTime,Load,PV,SOC,pricePerHour,interrupted load remain ,uninterrupted load remain)
        sampleTime,load,pv,pricePerHour,UnRemain,UnSwitch = self.state
        
        # 1.turn on switch 
        if action == 0 and UnRemain>0 and UnSwitch==0:
            self.uninterruptibleLoad.turn_on()
            reward.append(0.25)

        #2.  do nothing
        elif action == 1 : 
            pass

        # the uninterruptible Load operate itself
        self.uninterruptibleLoad.step()

        #calculate the cost at this sampletime (multiple 0.25 is for transforming pricePerHour  into per 15 min)
        if self.uninterruptibleLoad.switch == True:
            if (load + self.uninterruptibleLoad.AvgPowerConsume - pv) < 0:
                cost = 0 #encourage agent turn on loads when pv is high
            #PgridMax reward
            elif(load+self.uninterruptibleLoad.AvgPowerConsume-pv>self.PgridMax):
                reward.append(-1)
                cost = pricePerHour * 0.25 * self.uninterruptibleLoad.AvgPowerConsume

                #calculate cost and proportion
            else:

                cost = pricePerHour * 0.25 * self.uninterruptibleLoad.AvgPowerConsume    

        reward.append(-0.1*cost)

        if (sampleTime == 94) and (self.uninterruptibleLoad.getRemainDemand()!=0):
            reward.append(-10)
            
        #change to next state
        sampleTime = int(sampleTime+1)
        self.state=np.array([sampleTime,self.Load[sampleTime],self.PV[sampleTime],self.GridPrice[sampleTime],self.uninterruptibleLoad.getRemainDemand(),self.uninterruptibleLoad.switch])

        #check if all day is done
        done =  bool(sampleTime == 95)
        #REWARD


        reward = sum(reward)
        info = {'reward':reward}

        return self.state,reward,done,info

        
    def render(self):
        pass
    def reset(self):
        '''
        Starting State
        '''
        #each month pick one day for testing
        self.uninterruptibleLoad = WM(demand=3,executePeriod=5,AvgPowerConsume=1.5)

        self.i += 1
        self.Load = self.allLoad.iloc[:,self.i].tolist()

        #import PV
        if self.i  == 0:
            self.PV = self.allPV['Jan'].tolist()
        elif self.i  == 1:
            self.PV = self.allPV['Feb'].tolist()
        elif self.i  == 2:
            self.PV = self.allPV['Mar'].tolist()
        elif self.i  == 3:
            self.PV = self.allPV['Apr'].tolist()
        elif self.i  == 4:
            self.PV = self.allPV['May'].tolist()
        elif self.i  == 5:
            self.PV = self.allPV['Jun'].tolist()
        elif self.i  == 6:
            self.PV = self.allPV['July'].tolist()
        elif self.i  == 7:
            self.PV = self.allPV['Aug'].tolist()
        elif self.i  == 8:
            self.PV = self.allPV['Sep'].tolist()
        elif self.i  == 9:
            self.PV = self.allPV['Oct'].tolist()
        elif self.i  == 10:
            self.PV = self.allPV['Nov'].tolist()
        elif self.i  == 11:
            self.PV = self.allPV['Dec'].tolist()
        #reset state
        self.state=np.array([0,self.Load[0],self.PV[0],self.GridPrice[0],self.uninterruptibleLoad.demand,self.uninterruptibleLoad.switch])
        return self.state


if __name__ == '__main__':
    env = make("Hems-v9")
#     # Initialize episode
    states = env.reset()
    done = False
    step = 0
    Totalreward = 0
    while not done: # Episode timestep
        actions = env.action_space.sample()
        states, reward, done , info = env.step(action=actions)
        Totalreward += reward
    print(states)
        