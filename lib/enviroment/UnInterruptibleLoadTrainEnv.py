from lib.loads.uninterrupted import WM
from  gym import spaces
import numpy as np
from random import randint,uniform
from lib.enviroment.hemsTrainEnv import HemsEnv
from tensorforce import Environment

class UnIntEnv(HemsEnv):
    def __init__(self) :
        '''
        Action space
        observation space
        '''
        super().__init__()
        #import Base Parameter
        self.PgridMax = float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='PgridMax']['value'])[0])
        self.batteryCapacity=float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='batteryCapacity']['value'])[0])
#        self.uninterruptibleLoad = WM(demand=randint(1,20),executePeriod=randint(2,4),AvgPowerConsume=0.3)
        self.uninterruptibleLoad = WM(demand=randint(12,24),executePeriod=3,AvgPowerConsume=0.3)
        self.deltaSoc = [uniform(-0.15,0.15) for _ in range(96)]
        self.GridPrice = [uniform(1.73,6.2) for _ in range(96)]



    def states(self):
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
                #Delta SOC
                0.15,
                #Uninterruptible Remain
                75.0,
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
                #Delta SOC
                -0.15,
                #Uninterruptible Remain
                0.0,
                #Uninterruptible Switch
                0.0
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(lowerLimit,upperLimit,dtype=np.float32)        
        return dict(type='float', shape=self.observation_space.shape, min_value=lowerLimit,max_value=upperLimit)

    def actions(self):
        #action space
        return dict(type='int', num_values=2)

    def close(self):
        super().close()

    def reset(self):
        '''
        Starting State
        '''
        #pick one day from 360 days
        self.i = randint(1,359)
        self.Load = self.allLoad.iloc[:,self.i].tolist()
        if int( self.i / 30) == 0:
            self.PV = self.allPV['Jan'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif int(self.i / 30) == 1:
            self.PV = self.allPV['Feb'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif int(self.i / 30) == 2:
            self.PV = self.allPV['Mar'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif int(self.i / 30) == 3:
            self.PV = self.allPV['Apr'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif int(self.i / 30) == 4:
            self.PV = self.allPV['May'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif int(self.i / 30) == 5:
            self.PV = self.allPV['Jun'].tolist()
            self.GridPrice = self.summerGridPrice
        elif int(self.i / 30) == 6:
            self.PV = self.allPV['July'].tolist()
            self.GridPrice = self.summerGridPrice
        elif int(self.i / 30) == 7:
            self.PV = self.allPV['Aug'].tolist()
            self.GridPrice = self.summerGridPrice
        elif int(self.i / 30) == 8:
            self.PV = self.allPV['Sep'].tolist()
            self.GridPrice = self.summerGridPrice
        elif int(self.i / 30) == 9:
            self.PV = self.allPV['Oct'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif int(self.i / 30) == 10:
            self.PV = self.allPV['Nov'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif int(self.i / 30) == 11:
            self.PV = self.allPV['Dec'].tolist()
            self.GridPrice = self.notSummerGridPrice
        self.deltaSoc = [uniform(-0.15,0.15) for _ in range(96)]
        self.GridPrice = [uniform(1.73,6.2) for _ in range(96)]
#        self.uninterruptibleLoad = WM(demand=randint(1,20),executePeriod=randint(2,4),AvgPowerConsume=0.3)
        self.uninterruptibleLoad = WM(demand=randint(12,24),executePeriod=3,AvgPowerConsume=0.3)
        #reset state
        self.state=np.array([0,self.Load[0],self.PV[0],self.GridPrice[0],self.deltaSoc[0],self.uninterruptibleLoad.demand,self.uninterruptibleLoad.switch])
        #actions mask
        PgridMaxExceed = (self.Load[0]+self.deltaSoc[0]+self.uninterruptibleLoad.AvgPowerConsume-self.PV[0]) >= self.PgridMax

        self.action_mask = np.asarray([True,self.state[5]>0 and self.state[6]==False and not PgridMaxExceed])
        return self.state

    def execute(self,actions):
        '''
        interaction of each state(changes while taking actions)
        Rewards
        Episode Termination condition
        '''        
        #list for storing reward
        reward = []
        cost = 0
        #STATE (sampleTime,Load,PV,SOC,pricePerHour,Uninterruptible load remain ,uninterruptible load remain)
        sampleTime,load,pv,pricePerHour,deltaSoc,UnRemain,UnSwitch = self.state

        #  do nothing
        if actions == 0:
            pass
        #  turn on switch 
        elif actions == 1 : 
            self.uninterruptibleLoad.turn_on()

        # the uninterruptible Load operate itself
        self.uninterruptibleLoad.step()   
        # if the switch is on , calculate the electricity cost
        if self.uninterruptibleLoad.switch:
            Pess = deltaSoc*self.batteryCapacity
            if Pess<0:
                cost = (pricePerHour * 0.25 * (self.uninterruptibleLoad.AvgPowerConsume-pv+Pess))/self.uninterruptibleLoad.demand
            else:
                cost = (pricePerHour * 0.25 * (self.uninterruptibleLoad.AvgPowerConsume-pv))/self.uninterruptibleLoad.demand
        if cost<0:
            cost = 0 


        #reward
        reward.append(0.07-18*cost)
        if (sampleTime == 94) and (self.uninterruptibleLoad.getRemainDemand()!=0):
            reward.append(-10*self.uninterruptibleLoad.getRemainProcessPercentage())
        

        #change to next state
        sampleTime = int(sampleTime+1)
        self.state=np.array([sampleTime,self.Load[sampleTime],self.PV[sampleTime],self.GridPrice[sampleTime],self.deltaSoc[sampleTime],self.uninterruptibleLoad.getRemainDemand(),self.uninterruptibleLoad.switch])
        #actions mask
        PgridMaxExceed = (self.Load[sampleTime]+self.deltaSoc[sampleTime]+self.uninterruptibleLoad.AvgPowerConsume-self.PV[sampleTime]) >= self.PgridMax

        self.action_mask = np.asarray([True,self.state[5]>0 and self.state[6]==False and not PgridMaxExceed])
        #check if all day is done
        self.done =  bool(sampleTime == 95)
        #REWARD
        self.reward = sum(reward)
        states = dict(state=self.state,action_mask=self.action_mask)
        return states,self.done,self.reward

if __name__ == '__main__':
    from tensorforce import Agent
    environment = Environment.create(environment = UnIntEnv,max_episode_timesteps=96)
    agent = Agent.create(agent='/home/hems/LIU/RL_env/projects/RL_firstry/Load/UnInterruptible/Load_Agent/ppo.json', environment=environment)

    # Train for 100 episodes
    for _ in range(100):
        states = environment.reset()
        terminal = False
        while not terminal:
            actions = agent.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)
        