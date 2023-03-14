from  gym.envs.Hems.import_data import ImportData 
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
        self.initIndoorTemperature= float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='init_indoor_temperature(F)']['value'])[0])
        self.socInit = float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='SOCinit']['value'])[0])

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
        if self.i == 0:
            self.PV = self.allPV['Jan'].tolist()
        elif self.i == 1:
            self.PV = self.allPV['Feb'].tolist()
        elif self.i == 2:
            self.PV = self.allPV['Mar'].tolist()
        elif self.i == 3:
            self.PV = self.allPV['Apr'].tolist()
        elif self.i == 4:
            self.PV = self.allPV['May'].tolist()
        elif self.i == 5:
            self.PV = self.allPV['Jun'].tolist()
        elif self.i == 6:
            self.PV = self.allPV['July'].tolist()
        elif self.i == 7:
            self.PV = self.allPV['Aug'].tolist()
        elif self.i == 8:
            self.PV = self.allPV['Sep'].tolist()
        elif self.i == 9:
            self.PV = self.allPV['Oct'].tolist()
        elif self.i == 10:
            self.PV = self.allPV['Nov'].tolist()
        elif self.i == 11:
            self.PV = self.allPV['Dec'].tolist()

    #import Temperature
        self.allOutdoorTemperature = self.info.importTemperatureF()
        if self.i == 0:
            self.outdoorTemperature = self.allOutdoorTemperature['Jan'].tolist()
        elif self.i == 1:
            self.outdoorTemperature = self.allOutdoorTemperature['Feb'].tolist()
        elif self.i == 2:
            self.outdoorTemperature = self.allOutdoorTemperature['Mar'].tolist()
        elif self.i == 3:
            self.outdoorTemperature = self.allOutdoorTemperature['Apr'].tolist()
        elif self.i == 4:
            self.outdoorTemperature = self.allOutdoorTemperature['May'].tolist()
        elif self.i == 5:
            self.outdoorTemperature = self.allOutdoorTemperature['Jun'].tolist()
        elif self.i == 6:
            self.outdoorTemperature = self.allOutdoorTemperature['July'].tolist()
        elif self.i == 7:
            self.outdoorTemperature = self.allOutdoorTemperature['Aug'].tolist()
        elif self.i == 8:
            self.outdoorTemperature = self.allOutdoorTemperature['Sep'].tolist()
        elif self.i == 9:
            self.outdoorTemperature = self.allOutdoorTemperature['Oct'].tolist()
        elif self.i == 10:
            self.outdoorTemperature = self.allOutdoorTemperature['Nov'].tolist()
        elif self.i == 11:
            self.outdoorTemperature = self.allOutdoorTemperature['Dcb'].tolist()    
        
        #import User set Temperature
        self.allUserSetTemperature = self.info.importUserSetTemperatureF()
        if self.i == 0:
            self.userSetTemperature = self.allUserSetTemperature['Jan'].tolist()
        elif self.i == 1:
            self.userSetTemperature = self.allUserSetTemperature['Feb'].tolist()
        elif self.i == 2:
            self.userSetTemperature = self.allUserSetTemperature['Mar'].tolist()
        elif self.i == 3:
            self.userSetTemperature = self.allUserSetTemperature['Apr'].tolist()
        elif self.i == 4:
            self.userSetTemperature = self.allUserSetTemperature['May'].tolist()
        elif self.i == 5:
            self.userSetTemperature = self.allUserSetTemperature['Jun'].tolist()
        elif self.i == 6:
            self.userSetTemperature = self.allUserSetTemperature['July'].tolist()
        elif self.i == 7:
            self.userSetTemperature = self.allUserSetTemperature['Aug'].tolist()
        elif self.i == 8:
            self.userSetTemperature = self.allUserSetTemperature['Sep'].tolist()
        elif self.i == 9:
            self.userSetTemperature = self.allUserSetTemperature['Oct'].tolist()
        elif self.i == 10:
            self.userSetTemperature = self.allUserSetTemperature['Nov'].tolist()
        elif self.i == 11:
            self.userSetTemperature = self.allUserSetTemperature['Dcb'].tolist()

        #action we take (degree of charging/discharging power)
        self.action_space = spaces.Box(low=0,high=2,shape=(1,),dtype=np.float32)

        self.observation_space_name = np.array(['sampleTime', 'load', 'pv', 'SOC','pricePerHour','indoorTemperature','outdoorTemperature','userSetTemperature'])
        upperLimit = np.array(
            [
                #timeblock
                96,
                #load
                80,
                #PV
                20,
                #SOC
                1.0,
                #pricePerHour
                6,
                #indoor temperature
                104,
                #outdoor temperature
                104,
                #user set temperature
                104
            ],
            dtype=np.float32,
        )
        lowerLimit = np.array(
            [
                #timeblock
                0,
                #load
                0,
                #PV
                0,
                #SOC
                0.0,     
                #pricePerHour
                1,
                #indoor temperature
                35,
                #outdoor temperature
                50,
                #user set temperature
                35
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
        sampleTime,load,pv,soc,pricePerHour,indoorTemperature,outdoorTemperature,userSetTemperature = self.state



        sampleTime = int(sampleTime+1)

        #check if all day has done
        done = bool(
            sampleTime == 95
        )


        self.state=np.array([sampleTime,self.Load[sampleTime],self.PV[sampleTime],soc,self.GridPrice[sampleTime],indoorTemperature,self.outdoorTemperature[sampleTime],self.userSetTemperature[sampleTime]])



        #set placeholder for infomation
        info = {}
        reward = 0

        return self.state,reward,done,info

        
    def render(self):
        pass
    def reset(self):
        '''
        Starting State
        '''
        #each month pick one day for testing
        self.i += 1
        self.Load = self.allLoad.iloc[:,self.i].to_list()

    #setting PV
        if self.i == 0:
            self.PV = self.allPV['Jan'].tolist()
        elif self.i == 1:
            self.PV = self.allPV['Feb'].tolist()
        elif self.i == 2:
            self.PV = self.allPV['Mar'].tolist()
        elif self.i == 3:
            self.PV = self.allPV['Apr'].tolist()
        elif self.i == 4:
            self.PV = self.allPV['May'].tolist()
        elif self.i == 5:
            self.PV = self.allPV['Jun'].tolist()
        elif self.i == 6:
            self.PV = self.allPV['July'].tolist()
        elif self.i == 7:
            self.PV = self.allPV['Aug'].tolist()
        elif self.i == 8:
            self.PV = self.allPV['Sep'].tolist()
        elif self.i == 9:
            self.PV = self.allPV['Oct'].tolist()
        elif self.i == 10:
            self.PV = self.allPV['Nov'].tolist()
        elif self.i == 11:
            self.PV = self.allPV['Dec'].tolist()

    #setting Temperature
        if self.i == 0:
            self.outdoorTemperature = self.allOutdoorTemperature['Jan'].tolist()
        elif self.i == 1:
            self.outdoorTemperature = self.allOutdoorTemperature['Feb'].tolist()
        elif self.i == 2:
            self.outdoorTemperature = self.allOutdoorTemperature['Mar'].tolist()
        elif self.i == 3:
            self.outdoorTemperature = self.allOutdoorTemperature['Apr'].tolist()
        elif self.i == 4:
            self.outdoorTemperature = self.allOutdoorTemperature['May'].tolist()
        elif self.i == 5:
            self.outdoorTemperature = self.allOutdoorTemperature['Jun'].tolist()
        elif self.i == 6:
            self.outdoorTemperature = self.allOutdoorTemperature['July'].tolist()
        elif self.i == 7:
            self.outdoorTemperature = self.allOutdoorTemperature['Aug'].tolist()
        elif self.i == 8:
            self.outdoorTemperature = self.allOutdoorTemperature['Sep'].tolist()
        elif self.i == 9:
            self.outdoorTemperature = self.allOutdoorTemperature['Oct'].tolist()
        elif self.i == 10:
            self.outdoorTemperature = self.allOutdoorTemperature['Nov'].tolist()
        elif self.i == 11:
            self.outdoorTemperature = self.allOutdoorTemperature['Dcb'].tolist()

        self.allUserSetTemperature = self.info.importUserSetTemperatureF()
        if self.i == 0:
            self.userSetTemperature = self.allUserSetTemperature['Jan'].tolist()
        elif self.i == 1:
            self.userSetTemperature = self.allUserSetTemperature['Feb'].tolist()
        elif self.i == 2:
            self.userSetTemperature = self.allUserSetTemperature['Mar'].tolist()
        elif self.i == 3:
            self.userSetTemperature = self.allUserSetTemperature['Apr'].tolist()
        elif self.i == 4:
            self.userSetTemperature = self.allUserSetTemperature['May'].tolist()
        elif self.i == 5:
            self.userSetTemperature = self.allUserSetTemperature['Jun'].tolist()
        elif self.i == 6:
            self.userSetTemperature = self.allUserSetTemperature['July'].tolist()
        elif self.i == 7:
            self.userSetTemperature = self.allUserSetTemperature['Aug'].tolist()
        elif self.i == 8:
            self.userSetTemperature = self.allUserSetTemperature['Sep'].tolist()
        elif self.i == 9:
            self.userSetTemperature = self.allUserSetTemperature['Oct'].tolist()
        elif self.i == 10:
            self.userSetTemperature = self.allUserSetTemperature['Nov'].tolist()
        elif self.i == 11:
            self.userSetTemperature = self.allUserSetTemperature['Dcb'].tolist()

        #reset state
        self.state=np.array([0,self.Load[0],self.PV[0],self.socInit,self.GridPrice[0],self.initIndoorTemperature,self.outdoorTemperature[0],self.userSetTemperature[0]])
        return self.state


if __name__ == '__main__':
    env = make("Multi-hems-v0")
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
        