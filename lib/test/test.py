from tensorforce import Agent,Environment
from lib.loads.uninterrupted import WM
import pandas as pd

class Test():
    def __init__(self):
        self.testResult = {}
        for month in range(1,13):
            self.testResult[month] = pd.DataFrame()
    def uninterruptible(self):
        self.environment = Environment.create(environment='gym',level='Hems-v9')
        self.agent = Agent.load(directory = 'Load/UnInterruptible/saver_dir',environment=self.environment)
        wmObject = WM(demand=6,executePeriod=8,AvgPowerConsume=0.3)
        sampletime = []
        load = []
        pv = []
        price = []
        futurePrice = []
        switch = []
        unloadRemain = []
        totalReward = 0
        for month in range(1,13):
            states = self.environment.reset()
            sampletime.append(states[0])
            load.append(states[1])
            pv.append(states[2])
            price.append(states[3])
            futurePrice.append(states[4])
            unloadRemain.append(states[5])
            internals = self.agent.initial_internals()
            terminal = False
            while not terminal:
                actions, internals = self.agent.act(
                    states=states, internals=internals, independent=True, deterministic=True
                )
                states, terminal, reward = self.environment.execute(actions=actions)
                #1. switch on 
                if states[6] == 1: # washing machine's switch
                    switch.append(wmObject.AvgPowerConsume)#power
                #2. do nothing 
                else :
                    switch.append(0)

                sampletime.append(states[0])
                load.append(states[1])
                pv.append(states[2])
                price.append(states[3])
                futurePrice.append(states[4])
                unloadRemain.append(states[5])
                totalReward += reward
            switch.append(0)
            remain = [load[sampletime]-pv[sampletime] for sampletime in range(96)]
            self.testResult[month]['sampleTime'] = sampletime
            self.testResult[month]['remain'] = remain
            self.testResult[month]['price'] = price
            self.testResult[month]['futurePrice'] = futurePrice
            self.testResult[month]['unloadRemain'] = unloadRemain
            self.testResult[month]['switch'] = switch
            sampletime.clear()
            load.clear()
            pv.clear()
            price.clear()
            futurePrice.clear()
            switch.clear()
            unloadRemain.clear()
        print('Agent average episode reward: ', totalReward/12 ) 

if __name__ == '__main__':
    test = Test()
    print(test.testResult)