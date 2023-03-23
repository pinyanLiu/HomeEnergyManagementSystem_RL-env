from lib.Simulations.Simulation import Simulation
from tensorforce import Agent,Environment
from lib.enviroment.multiAgentEnv.multiAgentTestEnv import multiAgentTestEnv
from lib.plot.plot import Plot


class multiSimulation(Simulation):
    def __init__(self):
        super().__init__()
        self.environment = Environment.create(environment = multiAgentTestEnv,max_episode_timesteps=384)
        self.agent = Agent.load(directory = 'HLA/saver_dir',environment=self.environment)
    def simulation(self):
        sampletime = []
        load = []
        pv = []
        soc = []
        price = []
        deltaSoc = []
        indoorTemperature = []
        outdoorTemperature = []
        userSetTemperature = []
        intLoadRemain = []
        unLoadRemain = []        
        unLoadSwitch = []
        order = []
        Reward = []
        TotalReward = []
        totalReward = 0
        for month in range(12):
            states = self.environment.reset()
            totalState = self.environment.totalState
            sampletime.append(totalState["sampleTime"])
            load.append(totalState["fixLoad"])
            pv.append(totalState["PV"])
            soc.append(totalState["SOC"])
            price.append(totalState["pricePerHour"])
            deltaSoc.append(totalState["deltaSoc"])
            indoorTemperature.append(totalState["indoorTemperature"])
            outdoorTemperature.append(totalState["outdoorTemperature"])
            userSetTemperature.append(totalState["userSetTemperature"])
            intLoadRemain.append(totalState["intRemain"])
            unLoadRemain.append(totalState["unintRemain"])
            unLoadSwitch.append(totalState["unintSwitch"])
            order.append(totalState["order"])
            internals = self.agent.initial_internals()
            terminal = False
            while not terminal:
                #do action
                actions, internals = self.agent.act(
                    states=states, internals=internals, independent=True, deterministic=True
                )
                states, terminal, reward = self.environment.execute(actions=actions)
                #get total state information
                totalState = self.environment.totalState
                if totalState["order"]==3:
                    sampletime.append(totalState["sampleTime"])
                    load.append(totalState["fixLoad"])
                    pv.append(totalState["PV"])
                    soc.append(totalState["SOC"])
                    price.append(totalState["pricePerHour"])
                    deltaSoc.append(totalState["deltaSoc"])
                    indoorTemperature.append(totalState["indoorTemperature"])
                    outdoorTemperature.append(totalState["outdoorTemperature"])
                    userSetTemperature.append(totalState["userSetTemperature"])
                    intLoadRemain.append(totalState["intRemain"])
                    unLoadRemain.append(totalState["unintRemain"])
                    unLoadSwitch.append(totalState["unintSwitch"])
                    order.append(totalState["order"])
                self.totalReward.append(reward)
                Reward.append(reward)
                totalReward += reward
                Reward.append(0)
            print(len(load),len(pv),len(deltaSoc))
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
            soc.clear()
            price.clear()
            deltaSoc.clear()
            indoorTemperature.clear()
            outdoorTemperature.clear()
            userSetTemperature.clear()
            intLoadRemain.clear()
            unLoadRemain.clear()
            unLoadSwitch.clear()
            order.clear()
            Reward.clear()
        print('Agent average episode reward: ', sum(TotalReward)/len(TotalReward) ) 
        print('reward: ', TotalReward ) 
    
    def outputResult(self):
        output = Plot(self.testResult)
        output.remainPower()
        output.soc()
        output.indoorTemperature()
        output.outdoorTemperature()
        output.userSetTemperature()
        output.plotLoadPower()
        output.price()
        output.plotReward()
        output.plotResult('lib/plot/HRL')

    def getMean(self):
        return super().getMean()

    def getStd(self):
        return super().getStd()
        
    def __del__(self):
        return super().__del__()
