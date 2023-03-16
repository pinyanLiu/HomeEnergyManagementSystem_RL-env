class LLA():
    def __init__(self,environment,agent,mean,std) -> None:
        self.environment = environment
        self.agent = agent
        self.mean = mean 
        self.std = std
        self.state = []
        self.reward = 0

    def getState(self,allStates) -> None:
        pass


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
    def __init__(self, environment, agent, mean, std) -> None:
        super().__init__(environment, agent, mean, std)

    def getState(self, allStates) -> None:
        ##timeblock load PV SOC pricePerHour
        self.states.append(allStates['sampleTime'])
        self.states.append(allStates['fixLoad'])
        self.states.append(allStates['PV'])
        self.states.append(allStates['SOC'])
        self.states.append(allStates['pricePerHour'])

    def execute(self) -> None:
        self.actions, internals = self.agent.act(
                    states=self.states, internals=internals, independent=True, deterministic=True
                )
        self.state, terminal, self.reward = self.environment.execute(actions=self.actions,states = self.state) 

    def rewardStandardization(self) -> float:
        return super().rewardStandardization()
    
    def __del__(self):
        return super().__del__()

class hvacLLA(LLA):
    def __init__(self, environment, agent, mean, std) -> None:
        super().__init__(environment, agent, mean, std)

    def getState(self, allStates) -> None:
        #[timeblock,load,PV,pricePerHour,deltaSoc,indoor Temperature,outdoor temperature,user set temperature]
        self.states.append(allStates['sampleTime'])
        self.states.append(allStates['fixLoad'])
        self.states.append(allStates['PV'])
        self.states.append(allStates['pricePerHour'])
        self.states.append(allStates['deltaSoc'])
        self.states.append(allStates['indoorTemperature'])
        self.states.append(allStates['outdoorTemperature'])
        self.states.append(allStates['userSetTemperature'])

    def execute(self) -> None:
        self.actions, internals = self.agent.act(
                    states=self.states, internals=internals, independent=True, deterministic=True
                )
        self.state, terminal, self.reward = self.environment.execute(actions=self.actions,states = self.state) 

    def rewardStandardization(self) -> float:
        return super().rewardStandardization()
    
    def __del__(self):
        return super().__del__()

class intLLA(LLA):
    def __init__(self, environment, agent, mean, std ,Int) -> None:
        super().__init__(environment, agent, mean, std )
        self.interruptibleLoad = Int

    def getState(self, allStates) -> None:
        #[time block , load , PV ,pricePerHour , Delta SOC , interruptible Remain]
        self.states.append(allStates['sampleTime'])
        self.states.append(allStates['fixLoad'])
        self.states.append(allStates['PV'])
        self.states.append(allStates['pricePerHour'])
        self.states.append(allStates['deltaSoc'])
        self.states.append(allStates['intRemain'])

    def execute(self) -> None:
        self.actions, internals = self.agent.act(
                    states=self.states, internals=internals, independent=True, deterministic=True
                )
        self.state, terminal, self.reward = self.environment.execute(actions=self.actions,states = self.state , interruptiblLoad=self.interruptibleLoad) 

    def rewardStandardization(self) -> float:
        return super().rewardStandardization()
    
    def __del__(self):
        return super().__del__()

class unintLLA(LLA):
    def __init__(self, environment, agent, mean, std ,unInt) -> None:
        super().__init__(environment, agent, mean, std)
        self.uninterruptibleLoad = unInt

    def getState(self, allStates) -> None:
        #[time block , load , PV ,pricePerHour , Delta SOC , Uninterruptible Remain , Uninterruptible Switch]
        self.states.append(allStates['sampleTime'])
        self.states.append(allStates['fixLoad'])
        self.states.append(allStates['PV'])
        self.states.append(allStates['pricePerHour'])
        self.states.append(allStates['deltaSoc'])
        self.states.append(allStates['unintRemain'])
        self.states.append(allStates['unintSwitch'])

    def execute(self) -> None:
        self.actions, internals = self.agent.act(
                    states=self.states, internals=internals, independent=True, deterministic=True
                )
        self.state, terminal, self.reward = self.environment.execute(actions=self.actions,states = self.state , uninterruptibleLoad=self.uninterruptibleLoad) 

    def rewardStandardization(self) -> float:
        return super().rewardStandardization()
    
    def __del__(self):
        return super().__del__()        