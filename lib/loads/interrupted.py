class InterruptedLoad():
    def __init__(self,demand,AvgPowerConsume):
        self.switch = False
        self.AvgPowerConsume = AvgPowerConsume
        self.demand = demand
        self.alreadyTurnOn = 0
    def turn_on(self):
        #different kind of Interrupted Load may have different interact with action
        pass
    def turn_off(self):
        pass

    def getStatus(self):
        return(self.switch,self.AvgPowerConsume,self.demand,self.alreadyTurnOn)

    def getRemainDemand(self):
        return(self.demand-self.alreadyTurnOn)
    
    def getProcessPercentage(self):
        return ((self.alreadyTurnOn)/self.demand)

    def getRemainProcessPercentage(self):
        return -((self.alreadyTurnOn-self.demand)/self.demand)

    def reachDemand(self):
        return (self.alreadyTurnOn == self.demand)

    def getPowerConsume(self):
        return (self.alreadyTurnOn * self.AvgPowerConsume)
    
    def reset(self):
        self.switch = False
        self.alreadyTurnOn = 0

#AC is sub class of InterruptedLoad
class AC(InterruptedLoad):
    def __init__(self, demand = 8, AvgPowerConsume = 3000):
        super().__init__(demand, AvgPowerConsume)
    def turn_on(self):
        self.switch = True
        self.alreadyTurnOn += 1

    def turn_off(self):
        self.switch = False

    def getStatus(self):
        return super().getStatus()

    def getRemainDemand(self):
        return super().getRemainDemand()

    def getProcessPercentage(self):
        return super().getProcessPercentage()

    def getRemainProcessPercentage(self):
        return super().getRemainProcessPercentage()

    def reachDemand(self):
        return super().reachDemand()

    def getPowerConsume(self):
        return super().getPowerConsume()

    def reset(self):
        return super().reset()


if __name__ == '__main__':
    ac = AC(demand=8,AvgPowerConsume=420)
    for i in range (10):
        print('status',ac.getStatus())
        ac.turn_on()
        if ac.reachDemand():
            reward = ac.getRemainProcessPercentage()
        else:
            reward = ac.getProcessPercentage()
        #print('Percentage',ac.getProcessPercentage())
        #print('remainPercentage',ac.getRemainProcessPercentage())
        print("reward",reward)
        
    ac.reset()
    ac.turn_on()
    ac.turn_off()
    print(ac.getStatus())