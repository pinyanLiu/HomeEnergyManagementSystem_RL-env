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
        self.GridPrice = info.experimentData['GridPrice']
        self.PV = info.experimentData['PV']
        #pick one day from 360 days
        i = randint(1,360)
        self.Load = info.experimentData['Load'].iloc[:,i]
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

        #STATE
        sampleTime,load,pv,soc,price = self.state
        sampleTime = sampleTime+1


        #interaction
        #prevent the agent still want to charge while the battery is full of electricity
        if action == 0 and soc != 1:
            # 0. charging
            soc = soc+0.1
            #calculate the cost at this sampletime (multiple 0.25 is for transforming price per hour into per min)
            cost = price * 0.25 ( load + 0.1*self.BaseParameter['BaseParameter'].loc[self.BaseParameter['BaseParameter']['parameter_name']=='batteryCapacity',['value']] - pv  )

        #prevent the agent still want to discharge while the battery is lack of electricity
        elif action == 1 and soc != 0:
            # 1. discharging
            soc = soc-0.1
            #calculate the cost at this sampletime (multiple 0.25 is for transforming price per hour into per min)
            cost = price * 0.25 ( load - 0.1*self.BaseParameter['BaseParameter'].loc[self.BaseParameter['BaseParameter']['parameter_name']=='batteryCapacity',['value']] - pv  )

            # 2.stay
        else :
            #calculate the cost at this sampletime (multiple 0.25 is for transforming price per hour into per min)
            cost = price * 0.25 ( load - pv  )
            


            #check if all day is done
        if self.all_day_long <= 0:
            done = True
        else:
            done = False


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
        #reset state
        self.state = {'sample time':0,'grid price': self.info['grid price'][0],'PV': self.info['PV'][0],'SOC':0.2,'Pload':self.info['Pload'][0]}
        #reset time_block
        self.all_day_long = 96
        return np.array(self.state,dtype=np.float32)


if __name__ == '__main__':
    env = CemsEnvClass()
    env.action_space.sample()