from tensorforce import Environment,Agent
from lib.enviroment.multiAgentEnv.voidSocTestEnv import VoidSocTest
from lib.enviroment.multiAgentEnv.voidHvacTestEnv import VoidHvacTest
from lib.enviroment.multiAgentEnv.voidInterruptibleLoadTestEnv import VoidIntTest
from lib.enviroment.multiAgentEnv.voidUnInterruptibleLoadTestEnv import VoidUnIntTest
import numpy as np

class LLA():
    def __init__(self,mean,std,min,max) -> None:
        self.mean = mean 
        self.std = std
        self.min = min
        self.max = max
        self.states = []
        self.reward = 0


    def getState(self,allStates) -> None:
        pass

    def execute(self) -> None:
        pass 


    def rewardStandardization(self) -> None:
        self.reward =  (self.reward - self.mean)/self.std
    
    def rewardNormalization(self)-> None:
        self.reward = (self.reward-self.min)/(self.max-self.min)

    def __del__(self):
        pass

class socLLA(LLA):
    def __init__(self, mean, std,min,max,baseParameter) -> None:
        super().__init__(mean, std,min,max )
        self.environment = Environment.create(environment=VoidSocTest(baseParameter),max_episode_timesteps=96)
        self.agent = Agent.load(directory='Soc/saver_dir',environment=self.environment)
        self.internals = self.agent.initial_internals()

    def getState(self, allStates) -> None:
        ##timeblock load PV SOC pricePerHour
        self.states = np.array([allStates['sampleTime'],allStates['fixLoad'],allStates['PV'],allStates['SOC'],allStates['pricePerHour']],dtype=np.float32)

    def execute(self) -> None:
        self.actions, self.internals = self.agent.act(
                    states=self.states, internals=self.internals, independent=True, deterministic=True
                )
        self.states, terminal, self.reward = self.environment.execute(actions=self.actions) 


    def rewardStandardization(self):
        return super().rewardStandardization()
    
    def rewardNormalization(self) -> None:
        return super().rewardNormalization()
    
    def __del__(self):
        return super().__del__()

class hvacLLA(LLA):
    def __init__(self, mean, std,min,max , baseParameter , allOutdoorTemperature,allUserSetTemperature,id) -> None:
        super().__init__(mean, std,min,max)
        self.environment = Environment.create(environment=VoidHvacTest(baseParameter, allOutdoorTemperature,allUserSetTemperature),max_episode_timesteps=96)
        self.agent = Agent.load(directory='HVAC/saver_dir',environment=self.environment)
        self.internals = self.agent.initial_internals()
        self.id = id

    def getState(self, allStates) -> None:
        #[timeblock,load,PV,pricePerHour,deltaSoc,indoor Temperature,outdoor temperature,user set temperature]
        self.states = np.array([allStates['sampleTime'],allStates['fixLoad'],allStates['PV'],allStates['pricePerHour'],allStates['deltaSoc'],allStates['indoorTemperature'+str(self.id)],allStates['outdoorTemperature'],allStates['userSetTemperature'+str(self.id)]],dtype=np.float32)


    def execute(self) -> None:
        self.actions, self.internals = self.agent.act(
                    states=self.states, internals=self.internals, independent=True, deterministic=True
                )
        self.states, terminal, self.reward = self.environment.execute(actions=self.actions) 

    def rewardStandardization(self):
        return super().rewardStandardization()
    
    def rewardNormalization(self) -> None:
        return super().rewardNormalization()
    
    def __del__(self):
        return super().__del__()

class intLLA(LLA):
    def __init__(self, mean, std,min,max,baseParameter,Int,id) -> None:
        super().__init__(mean, std,min,max )
        self.interruptibleLoad = Int
        self.environment = Environment.create(environment=VoidIntTest(baseParameter,Int),max_episode_timesteps=96)
        self.agent = Agent.load(directory='Load/Interruptible/saver_dir',environment=self.environment)
        self.internals = self.agent.initial_internals()
        self.id = id
    def getState(self, allStates,actionMask) -> None:
        #[time block , load , PV ,pricePerHour , Delta SOC , interruptible Remain]
        self.states = dict(state=np.array([allStates['sampleTime'],allStates['fixLoad'],allStates['PV'],allStates['pricePerHour'],allStates['deltaSoc'],allStates['intRemain'+str(self.id)],allStates['intPreference'+str(self.id)]],dtype=np.float32),action_mask=actionMask)

    def execute(self) -> None:
        self.actions, self.internals = self.agent.act(
                    states=self.states, internals=self.internals, independent=True, deterministic=True
                )
        self.states, terminal, self.reward = self.environment.execute(actions=self.actions) 

    def rewardStandardization(self):
        return super().rewardStandardization()
    
    def rewardNormalization(self) -> None:
        return super().rewardNormalization()
    
    def __del__(self):
        return super().__del__()

class unintLLA(LLA):
    def __init__(self, mean, std,min,max,baseParameter ,unInt,id) -> None:
        super().__init__(mean, std,min,max )
        self.uninterruptibleLoad = unInt
        self.environment = Environment.create(environment=VoidUnIntTest(baseParameter,unInt),max_episode_timesteps=96)
        self.agent = Agent.load(directory='Load/UnInterruptible/saver_dir',environment=self.environment)
        self.internals = self.agent.initial_internals()
        self.id = id 

    def getState(self, allStates , actionMask) -> None:
        #[time block , load , PV ,pricePerHour , Delta SOC , Uninterruptible Remain , Uninterruptible Switch]
        self.states = dict(state=np.array([allStates['sampleTime'],allStates['fixLoad'],allStates['PV'],allStates['pricePerHour'],allStates['deltaSoc'],allStates['unintRemain'+str(self.id)],allStates['unintSwitch'+str(self.id)],allStates['unintPreference'+str(self.id)]],dtype=np.float32),action_mask=actionMask)


    def execute(self) -> None:
        self.actions, self.internals = self.agent.act(
                    states=self.states, internals=self.internals, independent=True, deterministic=True
                )
        self.states, terminal, self.reward = self.environment.execute(actions=self.actions) 

    def rewardStandardization(self):
        return super().rewardStandardization()
    
    def rewardNormalization(self) -> None:
        return super().rewardNormalization()
    
    def __del__(self):
        return super().__del__()        
    


