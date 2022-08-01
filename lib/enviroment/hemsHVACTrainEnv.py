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
        i = randint(1,359)
        #import Load 
        self.allLoad = self.info.importTrainingLoad()
        self.Load = self.allLoad.iloc[:,i].tolist()
        self.allPV = self.info.importPhotoVoltaic()
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

        
        #import Temperature
        self.allOutdoorTemperature = self.info.importTemperatureF()
        if int(i / 30) == 0:
            self.outdoorTemperature = self.allOutdoorTemperature['Jan'].tolist()
        elif int(i / 30) == 1:
            self.outdoorTemperature = self.allOutdoorTemperature['Feb'].tolist()
        elif int(i / 30) == 2:
            self.outdoorTemperature = self.allOutdoorTemperature['Mar'].tolist()
        elif int(i / 30) == 3:
            self.outdoorTemperature = self.allOutdoorTemperature['Apr'].tolist()
        elif int(i / 30) == 4:
            self.outdoorTemperature = self.allOutdoorTemperature['May'].tolist()
        elif int(i / 30) == 5:
            self.outdoorTemperature = self.allOutdoorTemperature['Jun'].tolist()
        elif int(i / 30) == 6:
            self.outdoorTemperature = self.allOutdoorTemperature['July'].tolist()
        elif int(i / 30) == 7:
            self.outdoorTemperature = self.allOutdoorTemperature['Aug'].tolist()
        elif int(i / 30) == 8:
            self.outdoorTemperature = self.allOutdoorTemperature['Sep'].tolist()
        elif int(i / 30) == 9:
            self.outdoorTemperature = self.allOutdoorTemperature['Oct'].tolist()
        elif int(i / 30) == 10:
            self.outdoorTemperature = self.allOutdoorTemperature['Nov'].tolist()
        elif int(i / 30) == 11:
            self.outdoorTemperature = self.allOutdoorTemperature['Dcb'].tolist()


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

        #calculate proportion
        cost = (load+Power_HVAC-pv)*pricePerHour
        proportion = Power_HVAC/(load+Power_HVAC-pv)
        cost= cost * proportion

        #REWARD
        reward = []
        #temperature reward
        if nextIndoorTemperature < (self.max_temperature+self.min_temperature)/2:
            r1 = 2*(nextIndoorTemperature-self.min_temperature)/(self.max_temperature-self.min_temperature)
        elif nextIndoorTemperature >= (self.max_temperature+self.min_temperature)/2:    
            r1 = -2*(nextIndoorTemperature-self.max_temperature)/(self.max_temperature-self.min_temperature)
        else:
            print("wtf are you doing?")
        if r1<-1:
            r1 = -0.8
        #cost reward
        r2 = -cost/5

        reward.append(r1)
        reward.append(r2)

        #change to next state
        sampleTime = int(sampleTime+1)

        #check if all day has done
        done = bool(
            sampleTime == 95
        )


        self.state=np.array([sampleTime,self.Load[sampleTime],self.PV[sampleTime],self.GridPrice[sampleTime],nextIndoorTemperature,self.outdoorTemperature[sampleTime]])



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

        #import Temperature
        if int(i / 30) == 0:
            self.outdoorTemperature = self.allOutdoorTemperature['Jan'].tolist()
        elif int(i / 30) == 1:
            self.outdoorTemperature = self.allOutdoorTemperature['Feb'].tolist()
        elif int(i / 30) == 2:
            self.outdoorTemperature = self.allOutdoorTemperature['Mar'].tolist()
        elif int(i / 30) == 3:
            self.outdoorTemperature = self.allOutdoorTemperature['Apr'].tolist()
        elif int(i / 30) == 4:
            self.outdoorTemperature = self.allOutdoorTemperature['May'].tolist()
        elif int(i / 30) == 5:
            self.outdoorTemperature = self.allOutdoorTemperature['Jun'].tolist()
        elif int(i / 30) == 6:
            self.outdoorTemperature = self.allOutdoorTemperature['July'].tolist()
        elif int(i / 30) == 7:
            self.outdoorTemperature = self.allOutdoorTemperature['Aug'].tolist()
        elif int(i / 30) == 8:
            self.outdoorTemperature = self.allOutdoorTemperature['Sep'].tolist()
        elif int(i / 30) == 9:
            self.outdoorTemperature = self.allOutdoorTemperature['Oct'].tolist()
        elif int(i / 30) == 10:
            self.outdoorTemperature = self.allOutdoorTemperature['Nov'].tolist()
        elif int(i / 30) == 11:
            self.outdoorTemperature = self.allOutdoorTemperature['Dcb'].tolist()



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