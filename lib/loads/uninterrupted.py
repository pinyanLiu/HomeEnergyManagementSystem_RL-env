class UninterruptedLoad():
    def __init__(self,demand,executePeriod,AvgPowerConsume):
        self.switch = False
        self.AvgPowerConsume = AvgPowerConsume
        self.demand = demand  # # How many times you want the load to execute
        self.executePeriod = executePeriod # unit : timeblock 
        self.alreadyTurnOn = 0  # for calculating the execute time in whole day
        self.alreadyTurnOnInPeriod = 0 # for calculating the execute time in per period

    def turn_on(self):
        pass

    def turn_off(self):
        pass
    
    def step(self):
        pass

    def getStatus(self):
        return(self.switch,self.AvgPowerConsume,self.demand,self.alreadyTurnOn,self.alreadyTurnOnInPeriod)

    def getRemainDemand(self):
        return(self.demand*self.executePeriod-self.alreadyTurnOn)

    def getProcessPercentage(self):
        return ((self.alreadyTurnOn)/self.demand*self.executePeriod)

    def getRemainProcessPercentage(self):
        return ((self.demand*self.executePeriod-self.alreadyTurnOn)/self.demand*self.executePeriod)

    def reachDemand(self):
        return (self.alreadyTurnOn >= self.demand*self.executePeriod)

    def reachExecutePeriod(self):
        return (self.alreadyTurnOnInPeriod == self.executePeriod)

    def getPowerConsume(self):
        return (self.alreadyTurnOn * self.AvgPowerConsume)
    
    def reset(self):
        self.switch = False
        self.alreadyTurnOn = 0
        self.alreadyTurnOnInPeriod = 0

#WM(washing machine) is sub class of uninterruptedLoad
class WM(UninterruptedLoad):
    def __init__(self, demand = 2,executePeriod = 3, AvgPowerConsume = 4.2):
        super().__init__(demand,executePeriod ,AvgPowerConsume)
    def turn_on(self):
        self.switch = True

    def turn_off(self):
        self.switch = False
    
    def step(self):
        if(self.switch == True):
            self.alreadyTurnOn += 1
            self.alreadyTurnOnInPeriod +=1
            if self.reachExecutePeriod():
                self.turn_off()
                self.alreadyTurnOnInPeriod = 0
        elif(self.switch == False):
            pass

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

    def reachExecutePeriod(self):
        return super().reachExecutePeriod()

    def getPowerConsume(self):
        return super().getPowerConsume()

    def reset(self):
        return super().reset()


if __name__ == '__main__':
    wm = WM(demand=3,executePeriod=3,AvgPowerConsume=0)
    for i in range (20):
        print('status',wm.getStatus(),i)
        if i == 3:
            wm.turn_on()
        if i == 9:
            wm.turn_on()
        if i == 15:
            wm.turn_on()
        wm.step()
        if wm.reachDemand():
            print('---reach demand---')
            
        if wm.reachExecutePeriod():
            print('---reach Period---')


    print(wm.getStatus())
