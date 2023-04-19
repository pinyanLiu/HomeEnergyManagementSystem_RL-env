from lib.Simulations.Simulation import Simulation
from tensorforce import Agent,Environment
from lib.enviroment.SocTestEnv import SocTest
from lib.plot.plot import Plot


class SocSimulation(Simulation):
    def __init__(self):
        super().__init__()
        self.environment = Environment.create(environment = SocTest,max_episode_timesteps=96)
        self.agent = Agent.load(directory = 'Soc/saver_dir',environment=self.environment)
        
    def simulation(self):
        sampletime = []
        load = []
        pv = []
        soc = []
        price = []
        deltaSoc = []
        Reward = []
        TotalReward = []
        totalReward = 0
        for month in range(12):
            states = self.environment.reset()
            sampletime.append(states[0])
            load.append(states[1])
            pv.append(states[2])
            soc.append(states[3])
            price.append(states[4])
            internals = self.agent.initial_internals()
            terminal = False
            while not terminal:
                actions, internals = self.agent.act(
                    states=states, internals=internals, independent=True, deterministic=True
                )
                states, terminal, reward = self.environment.execute(actions=actions)
                deltaSoc.append(actions[0])
                sampletime.append(states['state'][0])
                load.append(states['state'][1])
                pv.append(states['state'][2])
                soc.append(states['state'][3])
                price.append(states['state'][4])
                self.totalReward.append(reward)
                Reward.append(reward)
                totalReward += reward
            deltaSoc.append(0)
            Reward.append(0)
            remain = [load[sampletime]-pv[sampletime] for sampletime in range(96)]
            self.testResult[month]['sampleTime'] = sampletime
            self.testResult[month]['remain'] = remain
            self.testResult[month]['soc'] = soc
            self.testResult[month]['price'] = price
            self.testResult[month]['reward'] = Reward
            self.testResult[month]['deltaSoc']=deltaSoc
            TotalReward.append(totalReward)
            totalReward=0
            sampletime.clear()
            load.clear()
            pv.clear()
            soc.clear()
            price.clear()
            deltaSoc.clear()
            Reward.clear()
        print('Agent average episode reward: ', sum(TotalReward)/len(TotalReward) ) 
        print('reward: ', TotalReward ) 
    
    def outputResult(self):
        output = Plot(self.testResult)
        output.remainPower()
        output.price()
        output.soc()
        output.plotReward()
        output.plotResult('lib/plot/soc/')

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


