class InterruptedLoad():
    def __init__(self,demand,AvgPowerConsume):
        self.switch = False
        self.AvgPowerConsume = AvgPowerConsume
        self.demand = demand
        self.alreadyTurnOn = 0
    def step(self,action):
        #different kind of InterruptedLoad may have different interact with action
        pass

    def getStatus(self):
        return(self.switch,self.AvgPowerConsume,self.demand,self.alreadyTurnOn)

    def getRemainDemand(self):
        return(self.demand-self.alreadyTurnOn)

    def reachDemand(self):
        return (self.alreadyTurnOn >= self.demand)

    def getPowerConsume(self):
        return (self.alreadyTurnOn * self.AvgPowerConsume)
    
    def reset(self):
        self.switch = False
        self.alreadyTurnOn = 0

#AC is sub class of InterruptedLoad
class AC(InterruptedLoad):
    def __init__(self, demand, AvgPowerConsume):
        super().__init__(demand, AvgPowerConsume)
    def step(self, action):
        if action == 1:
            self.switch = True
            self.alreadyTurnOn += 1
        elif action == 2:
            self.switch = True
            self.alreadyTurnOn += 1
        else :
            self.switch = False

    def getStatus(self):
        return super().getStatus()

    def getRemainDemand(self):
        return super().getRemainDemand()

    def reachDemand(self):
        return super().reachDemand()

    def getPowerConsume(self):
        return super().getPowerConsume()

    def reset(self):
        return super().reset()


if __name__ == '__main__':
    ac = AC(demand=8,AvgPowerConsume=420)
    ac.step(action = 1)
    print(ac.getStatus())
    ac.step(action= 2)
    print(ac.getStatus())
    ac.step(action= 3)
    print(ac.getStatus())
    print(ac.getRemainDemand())
    print(ac.reachDemand())
    print(ac.getPowerConsume())
    ac.reset()
    print(ac.getStatus())
