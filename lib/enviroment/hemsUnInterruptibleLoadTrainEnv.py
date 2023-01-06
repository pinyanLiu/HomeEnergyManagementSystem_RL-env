from gym.envs.Hems.loads.uninterrupted import WM
from  gym import spaces
from gym import make
import numpy as np
from random import randint
from gym.envs.Hems.hemsTrainEnv import HemsEnv

class UnIntEnv(HemsEnv):
    def __init__(self) :
        '''
        Action space
        observation space
        '''
        super().__init__()
        #import Base Parameter
        self.PgridMax = float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='PgridMax']['value'])[0])
        
        self.uninterruptibleLoad = WM(demand=randint(1,4),executePeriod=randint(2,5),AvgPowerConsume=0.3)
        #action Uninterruptible load take (1.on 2.do nothing )
        self.action_space = spaces.Discrete(2)
        #self.observation_space_name = np.array(['sampleTime','load', 'pv', 'pricePerHour' ,'UnInterruptable Remain','switch'])
        #observation space 
        upperLimit = np.array(
            [
                #time block
                95,
                #load
                10.0,
                #PV
                10.0,
                #price per hour
                6.2,
                #Uninterruptible Remain
                20.0,
                #Uninterruptible Switch
                1.0
            ],
            dtype=np.float32,
        )
        lowerLimit = np.array(
            [
                #time block
                0.0,
                #load
                0.0,
                #PV
                0.0,
                #pricePerHour
                0.0,
                #Uninterruptible Remain
                0.0,
                #Uninterruptible Switch
                0.0
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(lowerLimit,upperLimit,dtype=np.float32)


    def step(self,action):
        '''
        interaction of each state(changes while taking action)
        Rewards
        Episode Termination condition
        '''
        #error message if getting wrong action
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action),err_msg
        
        #invalid action masking
        if self.action_mask[action]==False:
            return self.state,self.reward,self.done,self.info

        #list for storing reward
        reward = []
        cost = 0

        #STATE (sampleTime,Load,PV,SOC,pricePerHour,interrupted load remain ,uninterrupted load remain)
        #sampleTime,load,pv,pricePerHour,UnRemain,UnSwitch = self.state['states']
        sampleTime,load,pv,pricePerHour,UnRemain,UnSwitch = self.state
        # err_msg = f"{action}, {UnRemain}, {UnSwitch} invalid"
        # if action == 1:
        #     assert   UnRemain >=0 and UnSwitch == False ,err_msg
        
        #  do nothing
        if action == 0:
            pass
        #  turn on switch 
        elif action == 1 : 
            self.uninterruptibleLoad.turn_on()

        # the uninterruptible Load operate itself
        self.uninterruptibleLoad.step()   
        # if the switch is on , calculate the electricity cost
        if self.uninterruptibleLoad.switch:
            cost = (pricePerHour * 0.25 * self.uninterruptibleLoad.AvgPowerConsume) 


        #reward
        reward.append(-0.5*cost)
        if (sampleTime == 94) and (self.uninterruptibleLoad.getRemainDemand()!=0):
            reward.append(-0.8*self.uninterruptibleLoad.getRemainDemand())


        #change to next state
        sampleTime = int(sampleTime+1)
        self.state=np.array([sampleTime,self.Load[sampleTime],self.PV[sampleTime],self.GridPrice[sampleTime],self.uninterruptibleLoad.getRemainDemand(),self.uninterruptibleLoad.switch])
        #action mask
        self.action_mask = np.asarray([True,self.state[4]>0 and self.state[5]==False])
        #check if all day is done
        self.done =  bool(sampleTime == 95)
        #REWARD
        self.reward = sum(reward)


        self.info = reward

        return self.state,self.reward,self.done,self.info

        
    def render(self):
        pass
    def reset(self):
        '''
        Starting State
        '''
        super().reset()
        self.uninterruptibleLoad = WM(demand=randint(1,5),executePeriod=randint(2,4),AvgPowerConsume=0.3)


        #reset state
        self.state=np.array([0,self.Load[0],self.PV[0],self.GridPrice[0],self.uninterruptibleLoad.demand,self.uninterruptibleLoad.switch])
        #action mask
        self.action_mask = np.asarray([True,self.state[4]>0 and self.state[5]==False])
        return self.state


if __name__ == '__main__':
    env = make("Hems-v8")
#     # Initialize episode
    states = env.reset()
    done = False
    step = 0
    while not done: # Episode timestep
        actions = env.action_space.sample()
        states, reward, done , info = env.step(action=actions)
        print(info)
        