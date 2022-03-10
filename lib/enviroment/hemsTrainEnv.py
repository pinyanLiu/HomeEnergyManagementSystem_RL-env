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
        sampleTime,load,pv,soc,pricePerHour = self.state
        
        #interaction
        # if energy supply is greater than consumption
        if pv > load :
            cost = 0.001

        
        # if energy supply is less than consumption
        else:
                # 0. charging
                #prevent the agent still want to charge while the battery is full of electricity
            if action == 0 and (soc + 0.1) < 1:
                soc = soc+0.1
                #calculate the cost at this sampletime (multiple 0.25 is for transforming pricePerHour  into per 15 min)
                cost = pricePerHour * 0.25 *( load + 0.1*float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='batteryCapacity']['value'])[0]) - pv  )

                # 1. discharging
                #prevent the agent still want to discharge while the battery is lack of electricity
            elif action == 1 and (soc-0.1) >= 0:
                soc = soc-0.1

                #calculate the cost at this sampletime (multiple 0.25 is for transforming pricePerHour  into per 15 min)
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
            if (soc >= 1 and action == 0) or (soc <= 0 and action == 1) :
                reward.append(-2)
            # reward 1
            r1 = -cost/10000
            reward.append(r1)
            #reward 2
            if cost / (pricePerHour*0.25) >= 20000:
                reward.append(-5)
            else:    
                reward.append(0.0625)



        # if done
        else : 
            if (soc >= 1 and action == 0) or (soc <= 0 and action == 1) :
                reward.append(-2)
                print('im a dumb')
            # reward 1
            r1 = -cost/10000
            reward.append(r1)
            #reward 2
            if cost / (pricePerHour*0.25) >= 20000:
                reward.append(-5)
            else:    
                reward.append(0.0625)


            # reward 3
            if soc >= float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='SOCthreshold']['value'])[0]):
                r2 = 25
            else:
                r2 = -25   
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
        i = randint(0,359)
        self.Load = self.info.experimentData['Load'].iloc[:,i].tolist()

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
        self.state=np.array([0,self.Load[0],self.PV[0],float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='SOCinit']['value'])[0]),self.GridPrice[0]])
        return self.state


if __name__ == '__main__':
    env = make("Hems-v0")
#     # Initialize episode
    states = env.reset()
    done = False
    step = 0
    Totalreward = 0
    while not done: # Episode timestep
        actions = env.action_space.sample()
        states, reward, done , info = env.step(action=actions)
        Totalreward += reward
    print(Totalreward)
        