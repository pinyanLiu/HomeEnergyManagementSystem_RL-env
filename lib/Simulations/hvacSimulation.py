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
        indoorTemperature = []
        outdoorTemperature = []
        userSetTemperature = []
        hvac = []
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
            indoorTemperature.append(states[5])
            outdoorTemperature.append(states[6])
            userSetTemperature.append(states[7])
            internals = self.agent.initial_internals()
            terminal = False
            while not terminal:
                actions, internals = self.agent.act(
                    states=states, internals=internals, independent=True, deterministic=True
                )
                states, terminal, reward = self.environment.execute(actions=actions)
                hvac.append(actions[0])
                sampletime.append(states['state'][0])
                load.append(states['state'][1])
                pv.append(states['state'][2])
                price.append(states['state'][3])
                deltaSoc.append(states['state'][4])
                indoorTemperature.append(states['state'][5])
                outdoorTemperature.append(states['state'][6])
                userSetTemperature.append(states['state'][7])
                Reward.append(reward)
                totalReward += reward
            hvac.append(0)
            Reward.append(0)
            remain = [load[sampletime]-pv[sampletime]-deltaSoc[sampletime] for sampletime in range(96)]
            self.testResult[month]['sampleTime'] = sampletime
            self.testResult[month]['remain'] = remain
            self.testResult[month]['price'] = price
            self.testResult[month]['deltaSoc'] = deltaSoc
            self.testResult[month]['indoorTemperature'] = indoorTemperature
            self.testResult[month]['outdoorTemperature'] = outdoorTemperature
            self.testResult[month]['userSetTemperature'] = userSetTemperature
            self.testResult[month]['reward'] = Reward
            TotalReward.append(totalReward)
            totalReward=0
            sampletime.clear()
            load.clear()
            pv.clear()
            price.clear()
            deltaSoc.clear()
            indoorTemperature.clear()
            outdoorTemperature.clear()
            userSetTemperature.clear()
            Reward.clear()
        print('Agent average episode reward: ', sum(TotalReward)/len(TotalReward) ) 
        print('reward: ', TotalReward ) 
    
    def outputResult(self):
        output = Plot(self.testResult)
        output.remainPower()
        output.indoorTemperature()
        output.outdoorTemperature()
        output.userSetTemperature()
        output.price()
        output.plotReward()
        output.plotResult('lib/plot/hvac/')

    def __del__(self):
        # Close agent and environment
        if self.agent:
            self.agent.close()
        if self.environment:
            self.environment.close()
