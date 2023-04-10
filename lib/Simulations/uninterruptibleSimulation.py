from lib.Simulations.Simulation import Simulation
from tensorforce import Agent,Environment
from lib.enviroment.UnInterruptibleLoadTestEnv import UnIntTest
from lib.loads.uninterrupted import WM
import pandas as pd
from lib.plot.plot import Plot

class UnIntSimulation(Simulation):
    def __init__(self):
        super().__init__()
        self.environment = Environment.create(environment = UnIntTest,max_episode_timesteps=96)
        self.agent = Agent.load(directory = 'Load/UnInterruptible/saver_dir',environment=self.environment)
    def simulation(self):
        wmObject = WM(AvgPowerConsume=0.7)
        sampletime = []
        load = []
        pv = []
        price = []
        deltaSoc = []
        switch = []
        unloadRemain = []
        unintUserPreference = []
        Reward = []
        TotalReward = []
        totalReward = 0
        for month in range(12):
            states = self.environment.reset()
            sampletime.append(states[0])
            load.append(states[1])
            pv.append(states[2])
            price.append(states[3])
            deltaSoc.append(states[4])
            unloadRemain.append(states[5])
            unintUserPreference.append(states[7])
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
                unintUserPreference.append(states['state'][7])
                self.totalReward.append(reward)
                Reward.append(reward)
                totalReward += reward
            switch.append(0)
            Reward.append(0)
            remain = [load[sampletime]-pv[sampletime]-deltaSoc[sampletime] for sampletime in range(96)]
            self.testResult[month]['sampleTime'] = sampletime
            self.testResult[month]['remain'] = remain
            self.testResult[month]['price'] = price
            self.testResult[month]['deltaSoc'] = deltaSoc
            self.testResult[month]['unloadRemain'] = unloadRemain
            self.testResult[month]['switch'] = switch
            self.testResult[month]['reward'] = Reward
            self.testResult[month]['unintUserPreference'] = unintUserPreference

            TotalReward.append(totalReward)
            totalReward=0
            sampletime.clear()
            load.clear()
            pv.clear()
            price.clear()
            deltaSoc.clear()
            switch.clear()
            unloadRemain.clear()
            unintUserPreference.clear()
            Reward.clear()
        print('Agent average episode reward: ', sum(TotalReward)/len(TotalReward) ) 
        print('reward: ', TotalReward ) 
    
    def outputResult(self):
        output = Plot(self.testResult)
        output.remainPower()
        output.plotLoadPower()
        output.price()
        output.plotReward()
        output.plotUnintPreference()
        output.plotResult('lib/plot/uninterruptible/')


    def getMean(self):
        return super().getMean()

    def getStd(self):
        return super().getStd()

    def __del__(self):
        return super().__del__()
