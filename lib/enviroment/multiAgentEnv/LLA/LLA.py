from tensorforce import Environment,Agent
from lib.enviroment.multiAgentEnv.voidSocTestEnv import VoidSocTest
from lib.enviroment.multiAgentEnv.voidHvacTestEnv import VoidHvacTest
from lib.enviroment.multiAgentEnv.voidInterruptibleLoadTestEnv import VoidIntTest
from lib.enviroment.multiAgentEnv.voidUnInterruptibleLoadTestEnv import VoidUnIntTest
import numpy as np

class LLA():
    def __init__(self,mean,std) -> None:
        self.mean = mean 
        self.std = std
        self.states = []
        self.reward = 0


    def getState(self,allStates) -> None:
        self.states.clear()


    def execute(self) -> None:
        pass 


    def rewardStandardization(self) -> None:
        self.reward =  (self.reward - self.mean)/self.std

    def __del__(self):
        # Close agent and environment
        if self.agent:
            self.agent.close()
        if self.environment:
            self.environment.close()

class socLLA(LLA):
    def __init__(self, mean, std ,baseParameter) -> None:
        super().__init__(mean, std)
        self.environment = Environment.create(environment=VoidSocTest(baseParameter),max_episode_timesteps=96)
        self.agent = Agent.load(directory='Soc/saver_dir',environment=self.environment)

    def getState(self, allStates) -> None:
        ##timeblock load PV SOC pricePerHour
        self.states = np.array([allStates['sampleTime'],allStates['fixLoad'],allStates['PV'],allStates['SOC'],allStates['pricePerHour']])

    def execute(self) -> None:
        internals = self.agent.initial_internals()
        self.actions, internals = self.agent.act(
                    states=self.states, internals=internals, independent=True, deterministic=True
                )
        self.states, terminal, self.reward = self.environment.execute(actions=self.actions) 


    def rewardStandardization(self):
        return super().rewardStandardization()
    
    def __del__(self):
        return super().__del__()

class hvacLLA(LLA):
    def __init__(self, mean, std) -> None:
        super().__init__( mean, std)
        self.environment = Environment.create(environment=VoidHvacTest,max_episode_timesteps=96)
        self.agent = Agent.load(directory='HVAC/saver_dir',environment=self.environment)

    def getState(self, allStates) -> None:
        #[timeblock,load,PV,pricePerHour,deltaSoc,indoor Temperature,outdoor temperature,user set temperature]
        super().getState(allStates)

        self.states.append(allStates['sampleTime'])
        self.states.append(allStates['fixLoad'])
        self.states.append(allStates['PV'])
        self.states.append(allStates['pricePerHour'])
        self.states.append(allStates['deltaSoc'])
        self.states.append(allStates['indoorTemperature'])
        self.states.append(allStates['outdoorTemperature'])
        self.states.append(allStates['userSetTemperature'])

    def execute(self) -> None:
        internals = self.agent.initial_internals()
        self.actions, internals = self.agent.act(
                    states=self.states, internals=internals, independent=True, deterministic=True
                )
        self.states, terminal, self.reward = self.environment.execute(actions=self.actions) 

    def rewardStandardization(self):
        return super().rewardStandardization()
    
    def __del__(self):
        return super().__del__()

class intLLA(LLA):
    def __init__(self, mean, std ,Int) -> None:
        super().__init__( mean, std )
        self.interruptibleLoad = Int
        self.environment = Environment.create(environment=VoidIntTest,max_episode_timesteps=96)
        self.agent = Agent.load(directory='Load/Interruptible/saver_dir',environment=self.environment)

    def getState(self, allStates) -> None:
        #[time block , load , PV ,pricePerHour , Delta SOC , interruptible Remain]
        super().getState(allStates)
        self.states.append(allStates['sampleTime'])
        self.states.append(allStates['fixLoad'])
        self.states.append(allStates['PV'])
        self.states.append(allStates['pricePerHour'])
        self.states.append(allStates['deltaSoc'])
        self.states.append(allStates['intRemain'])

    def execute(self) -> None:
        internals = self.agent.initial_internals()
        self.actions, internals = self.agent.act(
                    states=self.states, internals=internals, independent=True, deterministic=True
                )
        self.states, terminal, self.reward = self.environment.execute(actions=self.actions , interruptiblLoad=self.interruptibleLoad) 

    def rewardStandardization(self):
        return super().rewardStandardization()
    
    def __del__(self):
        return super().__del__()

class unintLLA(LLA):
    def __init__(self, mean, std ,unInt) -> None:
        super().__init__( mean, std)
        self.uninterruptibleLoad = unInt
        self.environment = Environment.create(environment=VoidUnIntTest,max_episode_timesteps=96)
        self.agent = Agent.load(directory='Load/UnInterruptible/saver_dir',environment=self.environment)

    def getState(self, allStates) -> None:
        #[time block , load , PV ,pricePerHour , Delta SOC , Uninterruptible Remain , Uninterruptible Switch]
        super().getState(allStates)
        self.states.append(allStates['sampleTime'])
        self.states.append(allStates['fixLoad'])
        self.states.append(allStates['PV'])
        self.states.append(allStates['pricePerHour'])
        self.states.append(allStates['deltaSoc'])
        self.states.append(allStates['unintRemain'])
        self.states.append(allStates['unintSwitch'])

    def execute(self) -> None:
        internals = self.agent.initial_internals()
        self.actions, internals = self.agent.act(
                    states=self.states, internals=internals, independent=True, deterministic=True
                )
        self.states, terminal, self.reward = self.environment.execute(actions=self.actions , uninterruptibleLoad=self.uninterruptibleLoad) 

    def rewardStandardization(self):
        return super().rewardStandardization()
    
    def __del__(self):
        return super().__del__()        
    


