from gym.envs.Hems.loads.interrupted import AC
from gym.envs.Hems.hemsTrainEnv import HemsEnv
import numpy as np
from random import randint
from  gym import spaces


class UnIntEnv(HemsEnv):
    def __init__(self) :
        '''
        Action space
        observation space
        '''
        super().__init__()
        self.PgridMax = float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='PgridMax']['value'])[0])
        self.batteryCapacity = int(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='batteryCapacity']['value'])[0])


        self.interruptibleLoad = AC(demand=randint(1,49),AvgPowerConsume=1.5)
        # Interruptable load's actions  ( 1.on 2.off )
        self.action_space = spaces.Discrete(2)

        #observation space 
        upperLimit = np.array(
            [
                #timeblock
                95,
                #load
                10.0,
                #PV
                10.0,
                #pricePerHour
                6.0,
                #Interruptable Remain
                50.0,
            ],
            dtype=np.float32,
        )
        lowerLimit = np.array(
            [
                #timeblock
                0.0,
                #load
                0.0,
                #PV
                0.0,
                #pricePerHour
                0.0,
                #Interruptable Remain
                0.0,
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

        #list for storing reward
        reward = []
        cost = 0

        #STATE (sampleTime,Load,PV,SOC,pricePerHour,interrupted load remain)
        sampleTime,load,pv,pricePerHour,IntRemain, = self.state



        # 1. on 
        if action == 0 and IntRemain>0:
            self.interruptibleLoad.turn_on()
            reward.append(0.25)
            #calculate the cost at this sampletime (multiple 0.25 is for transforming pricePerHour  into per 15 min)
            if (load + self.interruptibleLoad.AvgPowerConsume - pv) < 0:
                cost = 0 #encourage agent turn on loads when pv is high
            #PgridMax reward
            elif(load+self.interruptibleLoad.AvgPowerConsume-pv>self.PgridMax):
                reward.append(-1)
                cost = pricePerHour * 0.25 * self.interruptibleLoad.AvgPowerConsume

            #calculate cost and proportion
            else:
                #proportion = np.abs(self.interruptibleLoad.AvgPowerConsume / (load + self.interruptibleLoad.AvgPowerConsume - pv) )
                #cost = proportion*(pricePerHour * 0.25 *( load + self.interruptibleLoad.AvgPowerConsume - pv ))  
                cost = pricePerHour * 0.25 * self.interruptibleLoad.AvgPowerConsume
        #2.  off
        elif action == 1 : 
            self.interruptibleLoad.turn_off()
            cost = 0
        # action = 0 and remain<=0
        else :
            reward.append(-1)

        reward.append(-0.1*cost)

        if (sampleTime == 94) and (self.interruptibleLoad.getRemainDemand()!=0):
            reward.append(-10)
            
        #change to next state
        sampleTime = int(sampleTime+1)
        self.state=np.array([sampleTime,self.Load[sampleTime],self.PV[sampleTime],self.GridPrice[sampleTime],self.interruptibleLoad.getRemainDemand()])

        #check if all day is done
        done =  bool(sampleTime == 95)
        #REWARD


        reward = sum(reward)
        info = {'reward':reward}

        return self.state,reward,done,info

        
    def render(self):
        pass
    def reset(self):
        '''
        Starting State
        '''
        super().reset()
        self.interruptibleLoad.reset()
        self.interruptibleLoad = AC(demand=randint(1,49),AvgPowerConsume=1.5)

        #reset state
        self.state=np.array([0,self.Load[0],self.PV[0],self.GridPrice[0],self.interruptibleLoad.demand])
        return self.state


if __name__ == '__main__':
    env = make("Hems-v4")
#     # Initialize episode
    states = env.reset()
    done = False
    step = 0
    while not done: # Episode timestep
        actions = env.action_space.sample()
        states, reward, done , info = env.step(action=actions)
        print(info)
        