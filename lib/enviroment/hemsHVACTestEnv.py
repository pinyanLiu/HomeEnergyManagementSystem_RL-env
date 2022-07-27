from  gym.envs.Hems.import_data import ImportData 
from  gym import Env
from  gym import spaces
from gym import make
import math
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

        self.info = ImportData(host= self.host ,user= self.user ,passwd= self.passwd ,db= self.db)
        
        self.BaseParameter = self.info.importBaseParameter()
        self.GridPrice = self.info.importGridPrice()
        self.GridPrice = self.GridPrice['price_value'].tolist()
        #pick one day from 360 days
        i = randint(0,359)
        self.Load = self.info.importTrainingLoad()
        self.Load = self.Load['Load'].iloc[:,i].tolist()
        self.PV = self.info.importPhotoVoltaic()

        if i / 12 == 0:
            self.PV = self.PV['Jan'].tolist()
        elif i / 12 == 1:
            self.PV = self.PV['Feb'].tolist()
        elif i / 12 == 2:
            self.PV = self.PV['Mar'].tolist()
        elif i / 12 == 3:
            self.PV = self.PV['Apr'].tolist()
        elif i / 12 == 4:
            self.PV = self.PV['May'].tolist()
        elif i / 12 == 5:
            self.PV = self.PV['Jun'].tolist()
        elif i / 12 == 6:
            self.PV = self.PV['July'].tolist()
        elif i / 12 == 7:
            self.PV = self.PV['Aug'].tolist()
        elif i / 12 == 8:
            self.PV = self.PV['Sep'].tolist()
        elif i / 12 == 9:
            self.PV = self.PV['Oct'].tolist()
        elif i / 12 == 10:
            self.PV = self.PV['Nov'].tolist()
        elif i / 12 == 11:
            self.PV = self.PV['Dec'].tolist()
        
        self.Temperature = self.info.importTemperature()
        if i / 12 == 0:
            self.Temperature = self.Temperature['Jan'].tolist()
        elif i / 12 == 1:
            self.Temperature = self.Temperature['Feb'].tolist()
        elif i / 12 == 2:
            self.Temperature = self.Temperature['Mar'].tolist()
        elif i / 12 == 3:
            self.Temperature = self.Temperature['Apr'].tolist()
        elif i / 12 == 4:
            self.Temperature = self.Temperature['May'].tolist()
        elif i / 12 == 5:
            self.Temperature = self.Temperature['Jun'].tolist()
        elif i / 12 == 6:
            self.Temperature = self.Temperature['July'].tolist()
        elif i / 12 == 7:
            self.Temperature = self.Temperature['Aug'].tolist()
        elif i / 12 == 8:
            self.Temperature = self.Temperature['Sep'].tolist()
        elif i / 12 == 9:
            self.Temperature = self.Temperature['Oct'].tolist()
        elif i / 12 == 10:
            self.Temperature = self.Temperature['Nov'].tolist()
        elif i / 12 == 11:
            self.Temperature = self.Temperature['Dcb'].tolist()
        #action we take (degree of charging/discharging power)
        self.action_space = spaces.Box(low=-0.1,high=0.1,shape=(1,),dtype=np.float32)

        #observation space ( Only SOC matters )
        self.observation_space_name = np.array(['sampleTime', 'load', 'pv', 'SOC', 'pricePerHour'])
        upperLimit = np.array(
            [
                #timeblock
                96,
                #load
                np.finfo(np.float32).max,
                #PV
                np.finfo(np.float32).max,
                #SOC
                self.BaseParameter.loc[self.BaseParameter['parameter_name']=='SOCmax','value'],
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
                #SOC
                self.BaseParameter.loc[self.BaseParameter['parameter_name']=='SOCmin','value'],         
                #pricePerHour
                np.finfo(np.float32).min,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(lowerLimit,upperLimit,dtype=np.float32)
        self.state = None
        self.cost = 0
        
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
        if (pv - soc_change*float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='batteryCapacity']['value'])[0])) >= load :
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
                cost = pricePerHour * 0.25 *( load + soc_change*float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='batteryCapacity']['value'])[0]) - pv  ) ## negative , because load < pv - soc_change
        
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
                cost = pricePerHour * 0.25 *( load + soc_change*float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='batteryCapacity']['value'])[0]) - pv  ) ## positive , because load > pv - soc_change

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
        self.cost = 0
        #each month pick one day for testing
        self.i += 1
        self.Load = self.info.experimentData['Load'].iloc[:,self.i]

        if self.i % 12 == 0:
            self.PV = self.info.experimentData['PV']['Jan'].tolist()
        elif self.i % 12 == 1:
            self.PV = self.info.experimentData['PV']['Feb'].tolist()
        elif self.i % 12 == 2:
            self.PV = self.info.experimentData['PV']['Mar'].tolist()
        elif self.i % 12 == 3:
            self.PV = self.info.experimentData['PV']['Apr'].tolist()
        elif self.i % 12 == 4:
            self.PV = self.info.experimentData['PV']['May'].tolist()
        elif self.i % 12 == 5:
            self.PV = self.info.experimentData['PV']['Jun'].tolist()
        elif self.i % 12 == 6:
            self.PV = self.info.experimentData['PV']['July'].tolist()
        elif self.i % 12 == 7:
            self.PV = self.info.experimentData['PV']['Aug'].tolist()
        elif self.i % 12 == 8:
            self.PV = self.info.experimentData['PV']['Sep'].tolist()
        elif self.i % 12 == 9:
            self.PV = self.info.experimentData['PV']['Oct'].tolist()
        elif self.i % 12 == 10:
            self.PV = self.info.experimentData['PV']['Nov'].tolist()
        elif self.i % 12 == 11:
            self.PV = self.info.experimentData['PV']['Dec'].tolist()
        #reset state
        self.state=np.array([0,self.Load[0],self.PV[0],float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='SOCinit']['value'])[0]),self.GridPrice[0]])
        return self.state


if __name__ == '__main__':
    env = make("Hems-v1")
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
        