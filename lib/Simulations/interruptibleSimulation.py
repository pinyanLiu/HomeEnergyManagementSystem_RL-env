from lib.Simulations.Simulation import Simulation
from tensorforce import Agent,Environment
from lib.enviroment.InterruptibleLoadTestEnv import IntTest
from lib.loads.interrupted import AC 
from lib.plot.plot import Plot

class IntSimulation(Simulation):
    def __init__(self):
        super().__init__()
        self.environment = Environment.create(environment = IntTest,max_episode_timesteps=96)
        self.agent = Agent.load(directory = 'Load/Interruptible/saver_dir',environment=self.environment)
    def simulation(self):
        acObject = AC(AvgPowerConsume=0.3)
        sampletime = []
        load = []
        pv = []
        price = []
        deltaSoc = []
        switch = []
        intloadRemain = []
        userPreference = []
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
            intloadRemain.append(states[5])
            internals = self.agent.initial_internals()
            terminal = False
            while not terminal:
                actions, internals = self.agent.act(
                    states=states, internals=internals, independent=True, deterministic=True
                )
                states, terminal, reward = self.environment.execute(actions=actions)

                #1. switch on 
                if actions == 1: # washing machine's switch
                    switch.append(acObject.AvgPowerConsume)#power
                #2. do nothing 
                else :
                    switch.append(0)

                sampletime.append(states['state'][0])
                load.append(states['state'][1])
                pv.append(states['state'][2])
                price.append(states['state'][3])
                deltaSoc.append(states['state'][4])
                intloadRemain.append(states['state'][5])
                userPreference.append(states['state'][6])
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
            self.testResult[month]['intloadRemain'] = intloadRemain
            self.testResult[month]['switch'] = switch
            self.testResult[month]['reward'] = Reward
            self.testResult[month]['userPreference'] = userPreference

            TotalReward.append(totalReward)
            totalReward=0
            sampletime.clear()
            load.clear()
            pv.clear()
            price.clear()
            deltaSoc.clear()
            switch.clear()
            intloadRemain.clear()
            Reward.clear()
            userPreference.clear()
        print('Agent average episode reward: ', sum(TotalReward)/len(TotalReward) ) 
        print('reward: ', TotalReward ) 
    
    def outputResult(self):
        output = Plot(self.testResult)
        output.remainPower()
        output.plotLoadPower()
        output.price()
        output.plotReward()
        output.plotPreference()
        output.plotResult('lib/plot/interruptible/')

    def getMean(self):
        return super().getMean()

    def getStd(self):
        return super().getStd()

    def __del__(self):
        return super().__del__()
