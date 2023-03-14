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
        self.actions, internals = self.agent.act(
                    states=self.states, internals=internals, independent=True, deterministic=True
                )
        self.state, terminal, self.reward = self.environment.execute(actions=self.actions) 


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
        self.states = [allStates[i] for i in [0,1,2,3,5]]

    def execute(self) -> None:
        return super().execute()


    def rewardStandardization(self) -> float:
        return super().rewardStandardization()
    
    def __del__(self):
        return super().__del__()

class hvacLLA(LLA):
    def __init__(self, environment, agent, mean, std) -> None:
        super().__init__(environment, agent, mean, std)

    def getState(self, allStates) -> None:
        #[timeblock,load,PV,pricePerHour,deltaSoc,indoor Temperature,outdoor temperature,user set temperature]
        self.states = [allStates[i] for i in [0,1,2,4,5,6,7,8]]

    def execute(self) -> None:
        return super().execute()

    def rewardStandardization(self) -> float:
        return super().rewardStandardization()
    
    def __del__(self):
        return super().__del__()

class intLLA(LLA):
    def __init__(self, environment, agent, mean, std) -> None:
        super().__init__(environment, agent, mean, std)

    def getState(self, allStates) -> None:
        #[time block , load , PV ,pricePerHour , Delta SOC , interruptible Remain]
        self.states = [allStates[i] for i in [0,1,2,4,5,9]]

    def execute(self) -> None:
        return super().execute()

    def rewardStandardization(self) -> float:
        return super().rewardStandardization()
    
    def __del__(self):
        return super().__del__()

class unintLLA(LLA):
    def __init__(self, environment, agent, mean, std) -> None:
        super().__init__(environment, agent, mean, std)

    def getState(self, allStates) -> None:
        #[time block , load , PV ,pricePerHour , Delta SOC , Uninterruptible Remain , Uninterruptible Switch]
        self.states = [allStates[i] for i in [0,1,2,4,5,10,11]]

    def execute(self) -> None:
        return super().execute()

    def rewardStandardization(self) -> float:
        return super().rewardStandardization()
    
    def __del__(self):
        return super().__del__()        