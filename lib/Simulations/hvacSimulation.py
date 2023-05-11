from lib.Simulations.Simulation import Simulation
from tensorforce import Agent,Environment
from lib.enviroment.HVACTestEnv import HvacTest
from lib.plot.plot import Plot


class HvacSimulation(Simulation):
    def __init__(self):
        super().__init__()
        self.environment = Environment.create(environment = HvacTest,max_episode_timesteps=96)
        self.agent = Agent.load(directory = 'HVAC/saver_dir',environment=self.environment)
    def simulation(self):
        sampletime = []
        load = []
        pv = []
        price = []
        deltaSoc = []
        indoorTemperature1 = []
        outdoorTemperature = []
        userSetTemperature1 = []
        ExceedPgridMaxTimes = []
        hvacPower1 = []
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
            indoorTemperature1.append(states[5])
            outdoorTemperature.append(states[6])
            userSetTemperature1.append(states[7])
            ExceedPgridMaxTimes.append(0)
            internals = self.agent.initial_internals()
            terminal = False
            while not terminal:
                actions, internals = self.agent.act(
                    states=states, internals=internals, independent=True, deterministic=True
                )
                states, terminal, reward = self.environment.execute(actions=actions)
                hvacPower1.append(actions[0])
                sampletime.append(states['state'][0])
                load.append(states['state'][1])
                pv.append(states['state'][2])
                price.append(states['state'][3])
                deltaSoc.append(states['state'][4])
                indoorTemperature1.append(states['state'][5])
                outdoorTemperature.append(states['state'][6])
                userSetTemperature1.append(states['state'][7])
                ExceedPgridMaxTimes.append(1 if actions[0]+states['state'][1]-states['state'][2]+states['state'][4]*10>10 else 0)
                self.totalReward.append(reward)
                Reward.append(reward)
                totalReward += reward
            hvacPower1.append(0)
            Reward.append(0)
            remain = [load[sampletime]-pv[sampletime]-deltaSoc[sampletime] for sampletime in range(96)]
            print(len(sampletime))
            print(len(remain))
            print(len(price))
            print(len(deltaSoc))
            self.testResult[month]['sampleTime'] = sampletime
            self.testResult[month]['remain'] = remain
            self.testResult[month]['price'] = price
            self.testResult[month]['deltaSoc'] = deltaSoc
            self.testResult[month]['indoorTemperature1'] = indoorTemperature1
            self.testResult[month]['outdoorTemperature'] = outdoorTemperature
            self.testResult[month]['userSetTemperature1'] = userSetTemperature1
            self.testResult[month]['hvacPower1'] = hvacPower1
            self.testResult[month]["ExceedPgridMaxTimes"] = ExceedPgridMaxTimes
            self.testResult[month]['reward'] = Reward
            TotalReward.append(totalReward)
            totalReward=0
            sampletime.clear()
            hvacPower1.clear()
            load.clear()
            pv.clear()
            price.clear()
            deltaSoc.clear()
            indoorTemperature1.clear()
            outdoorTemperature.clear()
            userSetTemperature1.clear()
            ExceedPgridMaxTimes.clear()
            Reward.clear()
        for month in range(12):
            print("month ",month, " ExceedPgridMaxTimes: ",sum(self.testResult[month]["ExceedPgridMaxTimes"]))
        print('Agent average episode reward: ', sum(TotalReward)/len(TotalReward) ) 
        print('reward: ', TotalReward ) 
    
    def outputResult(self):
        output = Plot(self.testResult)
        output.remainPower()
        output.indoorTemperature(1)
        output.outdoorTemperature()
        output.userSetTemperature()
        output.price()
        output.plotHVACPower(1)
        output.plotReward()
        output.plotResult('lib/plot/hvac/')

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
