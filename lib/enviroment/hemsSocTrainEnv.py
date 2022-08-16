from  gym.envs.Hems.import_data import ImportData 
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
        
        self.info = ImportData(host= self.host ,user= self.user ,passwd= self.passwd ,db= self.db)

        #import Base Parameter
        self.BaseParameter = self.info.importBaseParameter()
        self.batteryCapacity = int(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='batteryCapacity']['value'])[0])
        self.socInit = float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='SOCinit']['value'])[0])

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

        #action we take (degree of charging/discharging power)
        self.action_space = spaces.Box(low=-0.1,high=0.1,shape=(1,),dtype=np.float32)
        #observation space ( Only SOC matters )
        self.observation_space_name = np.array(['sampleTime', 'load', 'pv', 'SOC', 'pricePerHour'])
        upperLimit = np.array(
            [
                #timeblock
                96,
                #load
                80.0,
                #PV
                20.0,
                #SOC
                1.0,
                #pricePerHour
                6.0,
            ],
            dtype=np.float32,
        )
        lowerLimit = np.array(
            [
                #timeblock
                0,
                #load
                0.0,
                #PV
                0.0,
                #SOC
                0.0,         
                #pricePerHour
                1.0,
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
        #error message if getting action out of  boundary
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action),err_msg

    #STATE (sampleTime,Load,PV,SOC,pricePerHour)
        sampleTime,load,pv,soc,pricePerHour = self.state
        soc_change = float(action)
        # action(soc_change) is the degree of charging/discharging power .
        # if soc_change > 0 means charging , whereas soc_change < 0 means discharging.


    #interaction
        reward = []
        # if energy supply is greater than consumption means we don't have to buy grid , this should be encourage .
        if (pv + soc_change*self.batteryCapacity) >= load :
            if (soc + soc_change) < 0 :
                reward.append(-0.2)
                cost = 0.0001
            elif (soc + soc_change) > 1:
                reward.append(-0.2)
                cost = 0.0001

            else:
            #calculate the new soc for next state
                reward.append(0.1)
                soc = soc+soc_change
                cost = pricePerHour * 0.25 *( load + soc_change*self.batteryCapacity - pv  ) ## negative , because load < pv + soc_change
        
        # if energy supply is less than consumption
        else:
            #punish if the agent choose the action which shouldn't be choose(charge when SOC is full or discharge when SOC is null)
            if (soc + soc_change) < 0 :
                reward.append(-0.2)
                cost = 0.0001

            elif (soc + soc_change) > 1:
                reward.append(-0.2)
                cost = 0.0001

            else:
            #calculate the new soc for next state
                reward.append(0.1)
                soc = soc+soc_change
                cost = pricePerHour * 0.25 *( load + soc_change*self.batteryCapacity - pv  ) ## positive , because load > pv + soc_change

        #REWARD
      #  if sampleTime!=95:
        reward.append(-cost/10000)


        #change to next state
        sampleTime = int(sampleTime+1)

        #check if all day has done
        done = bool(
            sampleTime == 95
        )


        self.state=np.array([sampleTime,self.Load[sampleTime],self.PV[sampleTime],soc,self.GridPrice[sampleTime]])



        #set placeholder for infomation
        info = {'reward':reward}
        reward = sum(reward)

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
        #import PV
        if int(i / 30) == 0:
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
        states, reward, done , info = env.step(action=actions)
        print(info)
