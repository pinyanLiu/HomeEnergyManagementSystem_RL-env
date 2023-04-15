from lib.loads.interrupted import AC
from  gym import spaces
import numpy as np
from random import randint,uniform
from lib.enviroment.hemsTrainEnv import HemsEnv
from tensorforce import Environment

class IntEnv(HemsEnv):
    def __init__(self) :
        '''
        Action space
        observation space
        '''
        super().__init__()
        #import Base Parameter
        self.PgridMax = float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='PgridMax']['value'])[0])
        self.batteryCapacity=float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='batteryCapacity']['value'])[0])
        
        self.interruptibleLoad = AC(demand=randint(1,30),AvgPowerConsume=1.5)
        self.allIntPreference = self.info.importIntPreference()

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
                0.25,
                #interruptible Remain
                30.0,
                #user preference
                4
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
                -0.25,
                #interruptible Remain
                0.0,
                #user preference
                -1
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(lowerLimit,upperLimit,dtype=np.float32)
        return dict(type='float',shape=self.observation_space.shape,min_value=lowerLimit,max_value=upperLimit)

    def actions(self):
        #action space
        return dict(type='int',num_values=2)
    
    def close(self):
        super().close()

    def reset(self):
        '''
        Starting State
        '''
        #pick one day from 360 days
        self.i = randint(1,359)
        self.Load = self.allLoad.iloc[:,self.i].tolist()

        self.randomDeltaPrice  = [randint(-1,1) for _ in range(96)]
        self.randomDeltaPV = [uniform(-0.5,0.5) for _ in range(96)]
        self.randomDeltaSOC = [uniform(-0.05,0.05) for _ in range(96)]
        self.randomDeltaPreference = [randint(-1,1) for _ in range(96)]
        if int( self.i / 30) == 0:
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Jan'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
            self.deltaSOC = [min(max(x+y,-0.25),0.25) for x,y in zip(self.allDeltaSOC['Jan'].tolist(),self.randomDeltaSOC)]
            self.intUserPreference = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference['1'].tolist(),self.randomDeltaPreference)]
        elif int(self.i / 30) == 1:
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Feb'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
            self.deltaSOC = [min(max(x+y,-0.25),0.25) for x,y in zip(self.allDeltaSOC['Feb'].tolist(),self.randomDeltaSOC)]
            self.intUserPreference = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference['2'].tolist(),self.randomDeltaPreference)]
        elif int(self.i / 30) == 2:
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Mar'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
            self.deltaSOC = [min(max(x+y,-0.25),0.25) for x,y in zip(self.allDeltaSOC['Mar'].tolist(),self.randomDeltaSOC)]
            self.intUserPreference = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference['3'].tolist(),self.randomDeltaPreference)]
        elif int(self.i / 30) == 3:
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Apr'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
            self.deltaSOC = [min(max(x+y,-0.25),0.25) for x,y in zip(self.allDeltaSOC['Apr'].tolist(),self.randomDeltaSOC)]
            self.intUserPreference = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference['4'].tolist(),self.randomDeltaPreference)]
        elif int(self.i / 30) == 4:
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['May'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
            self.deltaSOC = [min(max(x+y,-0.25),0.25) for x,y in zip(self.allDeltaSOC['May'].tolist(),self.randomDeltaSOC)]
            self.intUserPreference = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference['5'].tolist(),self.randomDeltaPreference)]
        elif int(self.i / 30) == 5:
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Jun'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
            self.deltaSOC = [min(max(x+y,-0.25),0.25) for x,y in zip(self.allDeltaSOC['Jun'].tolist(),self.randomDeltaSOC)]
            self.intUserPreference = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference['6'].tolist(),self.randomDeltaPreference)]
        elif int(self.i / 30) == 6:
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['July'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.summerGridPrice,self.randomDeltaPrice) ]
            self.deltaSOC = [min(max(x+y,-0.25),0.25) for x,y in zip(self.allDeltaSOC['July'].tolist(),self.randomDeltaSOC)]
            self.intUserPreference = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference['7'].tolist(),self.randomDeltaPreference)]
        elif int(self.i / 30) == 7:
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Aug'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.summerGridPrice,self.randomDeltaPrice) ]
            self.deltaSOC = [min(max(x+y,-0.25),0.25) for x,y in zip(self.allDeltaSOC['Aug'].tolist(),self.randomDeltaSOC)]
            self.intUserPreference = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference['8'].tolist(),self.randomDeltaPreference)]
        elif int(self.i / 30) == 8:
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Sep'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.summerGridPrice,self.randomDeltaPrice) ]
            self.deltaSOC = [min(max(x+y,-0.25),0.25) for x,y in zip(self.allDeltaSOC['Sep'].tolist(),self.randomDeltaSOC)]
            self.intUserPreference = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference['9'].tolist(),self.randomDeltaPreference)]
        elif int(self.i / 30) == 9:
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Oct'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.summerGridPrice,self.randomDeltaPrice) ]
            self.deltaSOC = [min(max(x+y,-0.25),0.25) for x,y in zip(self.allDeltaSOC['Oct'].tolist(),self.randomDeltaSOC)]
            self.intUserPreference = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference['10'].tolist(),self.randomDeltaPreference)]
        elif int(self.i / 30) == 10:
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Nov'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
            self.deltaSOC = [min(max(x+y,-0.25),0.25) for x,y in zip(self.allDeltaSOC['Nov'].tolist(),self.randomDeltaSOC)]
            self.intUserPreference = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference['11'].tolist(),self.randomDeltaPreference)]
        elif int(self.i / 30) == 11:
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Dec'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
            self.deltaSOC = [min(max(x+y,-0.25),0.25) for x,y in zip(self.allDeltaSOC['Dcb'].tolist(),self.randomDeltaSOC)]
            self.intUserPreference = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference['12'].tolist(),self.randomDeltaPreference)]


        self.interruptibleLoad = AC(demand=randint(1,30),AvgPowerConsume=1.5)

        #reset state
        self.state=np.array([0,self.Load[0],self.PV[0],self.GridPrice[0],self.deltaSOC[0],self.interruptibleLoad.demand,self.intUserPreference[0]])
        #action mask
        PgridMaxExceed = (self.Load[0]+self.deltaSOC[0]*self.batteryCapacity+self.interruptibleLoad.AvgPowerConsume-self.PV[0]) >= self.PgridMax

        self.action_mask = np.asarray([True,self.state[5]>0 and not PgridMaxExceed])
        return self.state


    def execute(self,actions):
        '''
        interaction of each state(changes while taking action)
        Rewards
        Episode Termination condition
        '''
        #list for storing reward
        reward = []
        cost = 0
        #STATE (sampleTime,Load,PV,DeltaSOC,pricePerHour,interruptible load remain)
        sampleTime,load,pv,pricePerHour,deltaSoc,intRemain,intUserPreference = self.state
        # Turn off switch
        if actions == 0:
            self.interruptibleLoad.turn_off()
        #  turn on switch 
        elif actions == 1 : 
            self.interruptibleLoad.turn_on()

        self.interruptibleLoad.step()

        # if the switch is on , calculate the electricity cost
        if self.interruptibleLoad.switch:
            Pess = deltaSoc*self.batteryCapacity
            if Pess<0:
                cost = (pricePerHour * 0.25 * (self.interruptibleLoad.AvgPowerConsume-pv+Pess))/self.interruptibleLoad.demand
            else:
                cost = (pricePerHour * 0.25 * (self.interruptibleLoad.AvgPowerConsume-pv))/self.interruptibleLoad.demand
            reward.append(intUserPreference/5)#preference reward
        if cost<0:
            cost = 0 

        if(intUserPreference==4 and self.interruptibleLoad.switch==False):
            reward.append(-intUserPreference)

        #reward
        reward.append(0.07-9*cost)
        if (sampleTime == 94) :
            if(self.interruptibleLoad.getRemainDemand()!=0):
                reward.append(-20*self.interruptibleLoad.getRemainProcessPercentage())
            else:
                reward.append(10)
        #change to next state
        sampleTime = int(sampleTime+1)
        self.state=np.array([sampleTime,self.Load[sampleTime],self.PV[sampleTime],self.GridPrice[sampleTime],self.deltaSOC[sampleTime],self.interruptibleLoad.getRemainDemand(),self.intUserPreference[sampleTime]])

        #actions mask
        PgridMaxExceed = (self.Load[sampleTime]+self.deltaSOC[sampleTime]*self.batteryCapacity+self.interruptibleLoad.AvgPowerConsume-self.PV[sampleTime]) >= self.PgridMax
        
        self.action_mask = np.asarray([True,self.interruptibleLoad.getRemainDemand()>0 and not PgridMaxExceed])

        #check if all day is done
        self.done =  bool(sampleTime == 95)
        
        #REWARD
        self.reward = sum(reward)
        states = dict(state=self.state,action_mask=self.action_mask)

        return states,self.done,self.reward





if __name__ == '__main__':
    from tensorforce import Agent
    environment = Environment.create(environment = IntEnv,max_episode_timesteps=96)
    agent = Agent.create(agent='/home/hems/LIU/RL_env/projects/RL_firstry/Load/Interruptible/Load_Agent/ppo.json', environment=environment)

    # Train for 100 episodes
    for _ in range(100):
        states = environment.reset()
        terminal = False
        while not terminal:
            actions = agent.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)
        
        