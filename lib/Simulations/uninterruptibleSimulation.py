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
        sampletime = []
        load = []
        pv = []
        price = []
        deltaSoc = []
        unintSwitch = []
        unloadRemain = []
        unintUserPreference = []
        ExceedPgridMaxTimes=[]

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
            ExceedPgridMaxTimes.append(0)
            internals = self.agent.initial_internals()
            terminal = False
            while not terminal:
                actions, internals = self.agent.act(
                    states=states, internals=internals, independent=True, deterministic=True
                )
                states, terminal, reward = self.environment.execute(actions=actions)
                #1. unintSwitch on 
                if states['state'][6] == 1: # washing machine's unintSwitch
                    unintSwitch.append(self.environment.uninterruptibleLoad.AvgPowerConsume)#power
                #2. do nothing 
                else :
                    unintSwitch.append(0)

                sampletime.append(states['state'][0])
                load.append(states['state'][1])
                pv.append(states['state'][2])
                price.append(states['state'][3])
                deltaSoc.append(states['state'][4])
                unloadRemain.append(states['state'][5])
                unintUserPreference.append(states['state'][7])
                ExceedPgridMaxTimes.append(1 if states['state'][1]-states['state'][2]+states['state'][4]*10+states['state'][6]*self.environment.uninterruptibleLoad.AvgPowerConsume>10 else 0)
                self.totalReward.append(reward)
                Reward.append(reward)
                totalReward += reward
            unintSwitch.append(0)
            Reward.append(0)
            remain = [load[sampletime]-pv[sampletime]-deltaSoc[sampletime] for sampletime in range(96)]
            self.testResult[month]['sampleTime'] = sampletime
            self.testResult[month]['remain'] = remain
            self.testResult[month]['price'] = price
            self.testResult[month]['deltaSoc'] = deltaSoc
            self.testResult[month]['unloadRemain1'] = unloadRemain
            self.testResult[month]['unintSwitch1'] = unintSwitch
            self.testResult[month]['ExceedPgridMaxTimes'] = ExceedPgridMaxTimes
            self.testResult[month]['reward'] = Reward
            self.testResult[month]['unintUserPreference1'] = unintUserPreference

            TotalReward.append(totalReward)
            totalReward=0
            sampletime.clear()
            load.clear()
            pv.clear()
            price.clear()
            deltaSoc.clear()
            unintSwitch.clear()
            unloadRemain.clear()
            unintUserPreference.clear()
            ExceedPgridMaxTimes.clear()
            Reward.clear()
        for month in range(12):
            print("month ",month, " ExceedPgridMaxTimes: ",sum(self.testResult[month]["ExceedPgridMaxTimes"]))
        print('Agent average episode reward: ', sum(TotalReward)/len(TotalReward) ) 
        print('reward: ', TotalReward ) 
    
    def outputResult(self):
        output = Plot(self.testResult,single=True)
        output.remainPower(month=8)
        output.plotUnIntLoadPower(month=8)
        output.price(month=8)
        output.plotUnintPreference(month=8)
        output.plotResult('lib/plot/uninterruptible/')


    def getMean(self):
        return super().getMean()

    def getStd(self):
        return super().getStd()

    def getMax(self):
        return super().getMax()
    
    def getMin(self):
        return super().getMin()
    
    def __del__(self):
        return super().__del__()
