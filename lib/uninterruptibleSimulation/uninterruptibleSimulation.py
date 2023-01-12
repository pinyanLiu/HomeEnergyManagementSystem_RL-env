from tensorforce import Agent,Environment
from lib.enviroment.UnInterruptibleLoadTestEnv import UnIntTest
from lib.loads.uninterrupted import WM
import pandas as pd
class Test():
    def __init__(self):
        self.testResult = {}
        for month in range(12):
            self.testResult[month] = pd.DataFrame()
        self.environment = Environment.create(environment = UnIntTest,max_episode_timesteps=96)
        self.agent = Agent.load(directory = 'Load/UnInterruptible/saver_dir',environment=self.environment)
    def uninterruptible(self):
        wmObject = WM(demand=10,executePeriod=4,AvgPowerConsume=0.3)
        sampletime = []
        load = []
        pv = []
        price = []
        deltaSoc = []
        switch = []
        unloadRemain = []
        totalReward = 0
        for month in range(12):
            states = self.environment.reset()
            sampletime.append(states[0])
            load.append(states[1])
            pv.append(states[2])
            price.append(states[3])
            deltaSoc.append(states[4])
            unloadRemain.append(states[5])
            internals = self.agent.initial_internals()
            terminal = False
            while not terminal:
                actions, internals = self.agent.act(
                    states=states, internals=internals, independent=True, deterministic=True
                )
                states, terminal, reward = self.environment.execute(actions=actions)
                #1. switch on 
                if states['state'][6] == 1: # washing machine's switch
                    switch.append(wmObject.AvgPowerConsume)#power
                #2. do nothing 
                else :
                    switch.append(0)

                sampletime.append(states['state'][0])
                load.append(states['state'][1])
                pv.append(states['state'][2])
                price.append(states['state'][3])
                deltaSoc.append(states['state'][4])
                unloadRemain.append(states['state'][5])
                totalReward += reward
            switch.append(0)
            remain = [load[sampletime]-pv[sampletime] for sampletime in range(96)]
            self.testResult[month]['sampleTime'] = sampletime
            self.testResult[month]['remain'] = remain
            self.testResult[month]['price'] = price
            self.testResult[month]['deltaSoc'] = deltaSoc
            self.testResult[month]['unloadRemain'] = unloadRemain
            self.testResult[month]['switch'] = switch
            sampletime.clear()
            load.clear()
            pv.clear()
            price.clear()
            deltaSoc.clear()
            switch.clear()
            unloadRemain.clear()
        print('Agent average episode reward: ', totalReward/12 ) 


    def __del__(self):
        # Close agent and environment
        if self.agent:
            self.agent.close()
        if self.environment:
            self.environment.close()

if __name__ == '__main__':
    test = Test()
    print(test.testResult)