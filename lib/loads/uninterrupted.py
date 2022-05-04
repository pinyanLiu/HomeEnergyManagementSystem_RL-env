class UninterruptedLoad():
    def __init__(self,demand,AvgPowerConsume,executePeriod):
        self.switch = False
        self.AvgPowerConsume = AvgPowerConsume
        self.demand = demand  # unit : timeblock
        self.alreadyTurnOn = 0  # for calculating the execute time in whole day
        self.alreadyTurnOnInPeriod = 0 # for calculating the execute time in per period
        self.executePeriod = executePeriod # unit : timeblock 

    def turn_on(self):
        pass

    def turn_off(self):
        pass

    def getStatus(self):
        return(self.switch,self.AvgPowerConsume,self.demand,self.alreadyTurnOn,self.alreadyTurnOnInPeriod)

    def getRemainDemand(self):
        return(self.demand-self.alreadyTurnOn)

    def reachDemand(self):
        return (self.alreadyTurnOn >= self.demand)

    def reachTimeblockPerUse(self):
        return (self.alreadyTurnOnInPeriod >= self.executePeriod)

    def getPowerConsume(self):
        return (self.alreadyTurnOn * self.AvgPowerConsume)
    
    def reset(self):
        self.switch = False
        self.alreadyTurnOn = 0
        self.alreadyTurnOnInPeriod = 0

#WM(washing machine) is sub class of uninterruptedLoad
class WM(UninterruptedLoad):
    def __init__(self, demand = 12, AvgPowerConsume = 420 ,executePeriod = 6):
        super().__init__(demand, AvgPowerConsume,executePeriod)
    def turn_on(self):
        self.switch = True
        self.alreadyTurnOn += 1
        self.alreadyTurnOnInPeriod +=1
        if self.alreadyTurnOnInPeriod == self.executePeriod :
            self.alreadyTurnOnInPeriod = 0

    def turn_off(self):
        self.switch = False

    def getStatus(self):
        return super().getStatus()

    def getRemainDemand(self):
        return super().getRemainDemand()

    def reachDemand(self):
        return super().reachDemand()

    def reachTimeblockPerUse(self):
        return super().reachTimeblockPerUse()

    def getPowerConsume(self):
        return super().getPowerConsume()

    def reset(self):
        return super().reset()


if __name__ == '__main__':
    wm = WM(demand=8,AvgPowerConsume=420,executePeriod=4)
    for i in range (10):
        print('status',wm.getStatus())
        wm.turn_on()
        if wm.reachDemand:
            print('reach demand')
            
        if wm.reachTimeblockPerUse:
            print('reach Period')
        print('remaindemand',wm.getRemainDemand())
        print('powerConsume',wm.getPowerConsume())
    wm.reset()
    wm.turn_on()
    wm.turn_off()
    print(wm.getStatus())
