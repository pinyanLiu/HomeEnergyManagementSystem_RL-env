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
        self.info = ImportData(host= self.host ,user= self.user ,passwd= self.passwd ,db= self.db)

        #import Base Parameter
        self.BaseParameter = self.info.importBaseParameter()
        self.epsilon = float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='epsilon']['value'])[0])
        self.eta = float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='eta_HVAC']['value'])[0])
        self.A = float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='A(KW/F)']['value'])[0])
        self.max_temperature = float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='max_temperature(F)']['value'])[0])
        self.min_temperature = float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='min_temperature(F)']['value'])[0])
        self.initIndoorTemperature= float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='init_indoor_temperature(F)']['value'])[0])

        #import Grid price
        self.GridPrice = self.info.importGridPrice()
        self.GridPrice = self.GridPrice['price_value'].tolist()
        #pick one day from 360 days
        i = randint(1,360)
        #import Load 
        self.Load = self.info.importTrainingLoad()
        self.Load = self.Load.iloc[:,i].tolist()
        self.PV = self.info.importPhotoVoltaic()
        #import PV
        if int(i / 30) == 0:
            self.PV = self.PV['Jan'].tolist()
        elif int(i / 30) == 1:
            self.PV = self.PV['Feb'].tolist()
        elif int(i / 30) == 2:
            self.PV = self.PV['Mar'].tolist()
        elif int(i / 30) == 3:
            self.PV = self.PV['Apr'].tolist()
        elif int(i / 30) == 4:
            self.PV = self.PV['May'].tolist()
        elif int(i / 30) == 5:
            self.PV = self.PV['Jun'].tolist()
        elif int(i / 30) == 6:
            self.PV = self.PV['July'].tolist()
        elif int(i / 30) == 7:
            self.PV = self.PV['Aug'].tolist()
        elif int(i / 30) == 8:
            self.PV = self.PV['Sep'].tolist()
        elif int(i / 30) == 9:
            self.PV = self.PV['Oct'].tolist()
        elif int(i / 30) == 10:
            self.PV = self.PV['Nov'].tolist()
        elif int(i / 30) == 11:
            self.PV = self.PV['Dec'].tolist()

        
        #import Temperature
        self.outdoorTemperature = self.info.importTemperatureF()
        if int(i / 30) == 0:
            self.outdoorTemperature = self.outdoorTemperature['Jan'].tolist()
        elif int(i / 30) == 1:
            self.outdoorTemperature = self.outdoorTemperature['Feb'].tolist()
        elif int(i / 30) == 2:
            self.outdoorTemperature = self.outdoorTemperature['Mar'].tolist()
        elif int(i / 30) == 3:
            self.outdoorTemperature = self.outdoorTemperature['Apr'].tolist()
        elif int(i / 30) == 4:
            self.outdoorTemperature = self.outdoorTemperature['May'].tolist()
        elif int(i / 30) == 5:
            self.outdoorTemperature = self.outdoorTemperature['Jun'].tolist()
        elif int(i / 30) == 6:
            self.outdoorTemperature = self.outdoorTemperature['July'].tolist()
        elif int(i / 30) == 7:
            self.outdoorTemperature = self.outdoorTemperature['Aug'].tolist()
        elif int(i / 30) == 8:
            self.outdoorTemperature = self.outdoorTemperature['Sep'].tolist()
        elif int(i / 30) == 9:
            self.outdoorTemperature = self.outdoorTemperature['Oct'].tolist()
        elif int(i / 30) == 10:
            self.outdoorTemperature = self.outdoorTemperature['Nov'].tolist()
        elif int(i / 30) == 11:
            self.outdoorTemperature = self.outdoorTemperature['Dcb'].tolist()


        #action we take (degree of HVAC power)
        self.action_space = spaces.Box(low=0,high=2,shape=(1,),dtype=np.float32)
        #observation space 
        self.observation_space_name = np.array(['sampleTime', 'load', 'pv', 'pricePerHour','indoorTemperature','outdoorTemperature'])
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
                #indoor temperature
                self.max_temperature,
                #outdoor temperature
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
                #indoor temperature
                self.min_temperature,
                #outdoor temperature
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
        #error message if getting action out of  boundary
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action),err_msg

    #STATE (sampleTime,Load,PV,pricePerHour,indoor temperature ,outdoor temperature )
        sampleTime,load,pv,pricePerHour,indoorTemperature,outdoorTemperature = self.state
        Power_HVAC = float(action)


    #interaction

        #calculate the new indoor temperature for next state
        nextIndoorTemperature = self.epsilon*indoorTemperature+(1-self.epsilon)*(outdoorTemperature-(self.eta/self.A)*Power_HVAC)

        #calculate cost
        if pv >= load + Power_HVAC :
            cost = 0.0001
        else:
            cost = (load+Power_HVAC-pv)*pricePerHour

        #REWARD
        reward = []
        #temperature reward
        if nextIndoorTemperature < (self.max_temperature+self.min_temperature)/2:
            r1 = 2*(nextIndoorTemperature-self.min_temperature)/(self.max_temperature-self.min_temperature)
        elif nextIndoorTemperature >= (self.max_temperature+self.min_temperature)/2:    
            r1 = -2*(nextIndoorTemperature-self.max_temperature)/(self.max_temperature-self.min_temperature)
        else:
            print("wtf are you doing?")

        #cost reward
        r2 = -cost

        reward.append(r1)
        reward.append(r2)

        #change to next state
        sampleTime = int(sampleTime+1)

        #check if all day has done
        done = bool(
            sampleTime == 95
        )


        self.state=np.array([sampleTime,self.Load[sampleTime],self.PV[sampleTime],self.GridPrice[sampleTime],nextIndoorTemperature,outdoorTemperature])



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
        i = randint(1,360)
        self.Load = self.info.importTrainingLoad()
        self.Load = self.Load.iloc[:,i].tolist()
        self.PV = self.info.importPhotoVoltaic()
        #import PV
        if int(i / 30) == 0:
            self.PV = self.PV['Jan'].tolist()
        elif int(i / 30) == 1:
            self.PV = self.PV['Feb'].tolist()
        elif int(i / 30) == 2:
            self.PV = self.PV['Mar'].tolist()
        elif int(i / 30) == 3:
            self.PV = self.PV['Apr'].tolist()
        elif int(i / 30) == 4:
            self.PV = self.PV['May'].tolist()
        elif int(i / 30) == 5:
            self.PV = self.PV['Jun'].tolist()
        elif int(i / 30) == 6:
            self.PV = self.PV['July'].tolist()
        elif int(i / 30) == 7:
            self.PV = self.PV['Aug'].tolist()
        elif int(i / 30) == 8:
            self.PV = self.PV['Sep'].tolist()
        elif int(i / 30) == 9:
            self.PV = self.PV['Oct'].tolist()
        elif int(i / 30) == 10:
            self.PV = self.PV['Nov'].tolist()
        elif int(i / 30) == 11:
            self.PV = self.PV['Dec'].tolist()

    
        
        #import Temperature
        self.outdoorTemperature = self.info.importTemperatureF()
        if int(i / 30) == 0:
            self.outdoorTemperature = self.outdoorTemperature['Jan'].tolist()
        elif int(i / 30) == 1:
            self.outdoorTemperature = self.outdoorTemperature['Feb'].tolist()
        elif int(i / 30) == 2:
            self.outdoorTemperature = self.outdoorTemperature['Mar'].tolist()
        elif int(i / 30) == 3:
            self.outdoorTemperature = self.outdoorTemperature['Apr'].tolist()
        elif int(i / 30) == 4:
            self.outdoorTemperature = self.outdoorTemperature['May'].tolist()
        elif int(i / 30) == 5:
            self.outdoorTemperature = self.outdoorTemperature['Jun'].tolist()
        elif int(i / 30) == 6:
            self.outdoorTemperature = self.outdoorTemperature['July'].tolist()
        elif int(i / 30) == 7:
            self.outdoorTemperature = self.outdoorTemperature['Aug'].tolist()
        elif int(i / 30) == 8:
            self.outdoorTemperature = self.outdoorTemperature['Sep'].tolist()
        elif int(i / 30) == 9:
            self.outdoorTemperature = self.outdoorTemperature['Oct'].tolist()
        elif int(i / 30) == 10:
            self.outdoorTemperature = self.outdoorTemperature['Nov'].tolist()
        elif int(i / 30) == 11:
            self.outdoorTemperature = self.outdoorTemperature['Dcb'].tolist()



        #reset state
        self.state=np.array([0,self.Load[0],self.PV[0],self.GridPrice[0],self.initIndoorTemperature,self.outdoorTemperature[0]])
        return self.state



if __name__ == '__main__':
    env = make("Hems-v6")
#     # Initialize episode
    states = env.reset()
    done = False
    step = 0
    while not done: # Episode timestep
        actions = env.action_space.sample()
        states, reward, done , info = env.step(action=actions)
        print(info)