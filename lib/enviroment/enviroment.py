import gym
from  gym import Env
from  gym import spaces
import numpy as np
from random import randint
class HemsEnv(Env):
    def __init__(self,info) :
        '''
        Action space
        observation space
        '''
        self.info = info
        self.BaseParameter = info.experimentData['BaseParameter']
        self.GridPrice = info.experimentData['GridPrice'].loc[:,['price_value']].tolist()
        self.PV = info.experimentData['PV'].tolist()
        #pick one day from 360 days
        i = randint(1,360)
        self.Load = info.experimentData['Load'].iloc[:,i].tolist()
        #action we take (charge , discharge , stay)
        self.action_space = spaces.Discrete(3)
        #observation space
        upperLimit = np.array(
            [
                self.BaseParameter['BaseParameter'].loc[self.BaseParameter['BaseParameter']['parameter_name']=='SOCmax',['value']],
                self.BaseParameter['BaseParameter'].loc[self.BaseParameter['BaseParameter']['parameter_name']=='PowerMaxCharge',['value']],
            ],
            dtype=np.float32,
        )
        lowerLimit = np.array(
            [
                self.BaseParameter['BaseParameter'].loc[self.BaseParameter['BaseParameter']['parameter_name']=='SOCmin',['value']],
                self.BaseParameter['BaseParameter'].loc[self.BaseParameter['BaseParameter']['parameter_name']=='PowerMaxdisCharge',['value']],
            ],
            dtype=np.float32,
        )
        self.observationSpace = spaces.Box(lowerLimit,upperLimit,dtype=np.float32)
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
        #prevent the agent still want to charge while the battery is full of electricity
        if action == 0 and soc != 1:
            # 0. charging
            soc = soc+0.1
            #calculate the cost at this sampletime (multiple 0.25 is for transforming pricePerHour  into per min)
            cost = pricePerHour * 0.25 ( load + 0.1*self.BaseParameter['BaseParameter'].loc[self.BaseParameter['BaseParameter']['parameter_name']=='batteryCapacity',['value']] - pv  )

        #prevent the agent still want to discharge while the battery is lack of electricity
        elif action == 1 and soc != 0:
            # 1. discharging
            soc = soc-0.1
            #calculate the cost at this sampletime (multiple 0.25 is for transforming pricePerHour  into per min)
            cost = pricePerHour * 0.25 ( load - 0.1*self.BaseParameter['BaseParameter'].loc[self.BaseParameter['BaseParameter']['parameter_name']=='batteryCapacity',['value']] - pv  )

            # 2.stay
        else :
            #calculate the cost at this sampletime (multiple 0.25 is for transforming pricePerHour  into per min)
            cost = pricePerHour * 0.25 ( load - pv  )
            
        #change to next state
        sampleTime = sampleTime+1
        self.state = {'sampleTime':sampleTime,'load': self.Load[self.state['sampleTime']],'pv': self.PV[self.state['sampleTime']],'SOC':soc,'pricePerHour':self.GridPrice[self.state['sampleTime']]}

        #check if all day is done
        done = bool(
            sampleTime ==95
        )

        #REWARD
        reward = []
        if not done:
            if soc == 1 and action == 1:
                reward.append(-1)
            elif soc == 0 and action == -1 :
                reward.append(-1)



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
        i = randint(1,360)
        self.Load = self.info.experimentData['Load'].iloc[:,i]
        #reset state
        self.state = {'sampleTime':0,'load': self.Load[self.state['sampleTime']],'pv': self.PV[self.state['sampleTime']],'SOC':self.BaseParameter['BaseParameter'].loc[self.BaseParameter['BaseParameter']['parameter_name']=='SOCinit',['value']],'pricePerHour':self.GridPrice[self.state['sampleTime']]}
        return self.state


if __name__ == '__main__':
    env = CemsEnvClass()
    env.action_space.sample()