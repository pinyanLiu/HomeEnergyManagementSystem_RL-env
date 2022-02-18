from  gym.envs.enviroment.import_data import ImportData 
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
        with open("mysqlData.yaml","r") as f:
            self.mysqlData = load(f,SafeLoader)

        self.host = self.mysqlData['host']
        self.user = self.mysqlData['user']
        self.passwd = self.mysqlData['passwd']
        self.db = self.mysqlData['db']
        self.info = ImportData(host= self.host ,user= self.user ,passwd= self.passwd ,db= self.db)
        
        self.BaseParameter = self.info.experimentData['BaseParameter']
        self.GridPrice = self.info.experimentData['GridPrice']['price_value'].tolist()
        self.PV = self.info.experimentData['PV']['Jan'].tolist()
        #pick one day from 360 days
        i = randint(1,360)
        self.Load = self.info.experimentData['Load'].iloc[:,i].tolist()
        #action we take (charge , discharge , stay)
        self.action_space = spaces.Discrete(3)
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
        
    def step(self,action):
        '''
        interaction of each state(changes while taking action)
        Rewards
        Episode Termination condition
        '''
        #error message if getting wrong action
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action),err_msg

        #STATE (sampleTime,Load,PV,SOC,pricePerHour)
        
        # sampleTime = self.state['sampleTime']
        # load = self.state['load']
        # pv = self.state['pv']
        # soc = self.state['SOC']
        # pricePerHour = self.state['pricePerHour']

        sampleTime,load,pv,soc,pricePerHour = self.state

        #interaction

        # if energy supply is greater than consumption
        if pv > load and (soc + 0.1) < 1:
            soc = soc + 0.1
            cost = 0
        elif pv >load and (soc+0.1) >=1 :
            cost = 0
        
        # if energy supply is less than consumption
        else:
                # 0. charging
                #prevent the agent still want to charge while the battery is full of electricity
            if action == 0 and (soc + 0.1) < 1:
                soc = soc+0.1
                #calculate the cost at this sampletime (multiple 0.25 is for transforming pricePerHour  into per min)
                cost = pricePerHour * 0.25 *( load + 0.1*float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='batteryCapacity']['value'])[0]) - pv  )

                # 1. discharging
                #prevent the agent still want to discharge while the battery is lack of electricity
            elif action == 1 and (soc-0.1) >= 0:
                soc = soc-0.1
                #calculate the cost at this sampletime (multiple 0.25 is for transforming pricePerHour  into per min)
                cost = pricePerHour * 0.25 *( load - 0.1*float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='batteryCapacity']['value'])[0]) - pv  )

                # 2.stay
            else :
                #calculate the cost at this sampletime (multiple 0.25 is for transforming pricePerHour  into per min)
                cost = pricePerHour * 0.25 *( load - pv  )
            
        #change to next state
        sampleTime = int(sampleTime+1)
        #self.state = {'sampleTime':sampleTime,'load': self.Load[self.state['sampleTime']],'pv': self.PV[self.state['sampleTime']],'SOC':soc,'pricePerHour':self.GridPrice[self.state['sampleTime']]}
        self.state=np.array([sampleTime,self.Load[sampleTime],self.PV[sampleTime],soc,self.GridPrice[sampleTime]])
        #check if all day is done
        done = bool(
            sampleTime == 95
        )

        #REWARD
        reward = []
        if not done:
            #punish if the agent choose the action which shouldn't be choose(charge when SOC is full or discharge when SOC is null)
            if (soc == 1 and action == 1) or (soc == 0 and action == -1) :
                reward.append(-1)
            # reward 1
            r1 = -cost
            reward.append(r1)
            
        else : 
            if (soc == 1 and action == 1) or (soc == 0 and action == -1) :
                reward.append(-1)
            # reward 1
            r1 = -cost
            reward.append(r1)
            # reward 2
            if soc >= float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='SOCthreshold']['value'])[0]):
                r2 = 1000
            else:
                r2 = -1000    
            reward.append(r2)


        reward = sum(reward)



        #set placeholder for infomation
        info = {}

        return self.state,reward,done,info

        
    def render(self):
        pass
    def reset(self):
        '''
        Starting State
        '''

        #pick one day from 360 days
        i = randint(1,359)
        self.Load = self.info.experimentData['Load'].iloc[:,i]
        #reset state
        #self.state = {'sampleTime':0,'load': self.Load[0],'pv': self.PV[0],'SOC': float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='SOCinit']['value'])[0]),'pricePerHour':self.GridPrice[0]}
        self.state=np.array([0,self.Load[0],self.PV[0],float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='SOCinit']['value'])[0]),self.GridPrice[0]])
        return self.state


if __name__ == '__main__':
    env = make("Hems-v0")
#     # Initialize episode
    states = env.reset()
    done = False
    step = 0
    while not done: # Episode timestep
        print(states)
        actions = env.action_space.sample()
        states, reward, done , info = env.step(action=actions)
        