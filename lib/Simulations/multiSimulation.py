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
        remain = []
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
        intUserPreference = []
        unintUserPreference = []
        intSwitch = []
        unintSwitch = []
        order = []
        Reward = []
        TotalReward = []
        totalReward = 0
        for month in range(12):
            states = self.environment.reset()
            totalState = self.environment.totalState
            # sampletime.append(totalState["sampleTime"])
            # remain.append(states[1])
            # load.append(totalState["fixLoad"])
            # pv.append(totalState["PV"])
            # soc.append(totalState["SOC"])
            # price.append(totalState["pricePerHour"])
            # deltaSoc.append(totalState["deltaSoc"])
            # indoorTemperature.append(totalState["indoorTemperature"])
            # outdoorTemperature.append(totalState["outdoorTemperature"])
            # userSetTemperature.append(totalState["userSetTemperature"])
            # intLoadRemain.append(totalState["intRemain"])
            # unLoadRemain.append(totalState["unintRemain"])
            # intUserPreference.append(totalState["intUserPreference"])
            #unintUserPreference.append(totalState["unintPreference"])
            # intSwitch.append(totalState["intSwitch"])
            # unintSwitch.append(totalState["unintSwitch"])
            # order.append(totalState["order"])
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
                    remain.append(states['state'][2])
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
                    intUserPreference.append(totalState["intPreference"])
                    unintUserPreference.append(totalState["unintPreference"])
                    intSwitch.append(totalState["intSwitch"])
                    unintSwitch.append(totalState["unintSwitch"])
                    order.append(totalState["order"])
                    Reward.append(reward)
                totalReward += reward
            self.testResult[month]['sampleTime'] = sampletime
            self.testResult[month]['remain'] = remain
            self.testResult[month]['price'] = price
            self.testResult[month]['soc'] = soc
            self.testResult[month]['deltaSoc'] = deltaSoc
            self.testResult[month]['indoorTemperature'] = indoorTemperature
            self.testResult[month]['outdoorTemperature'] = outdoorTemperature
            self.testResult[month]['userSetTemperature'] = userSetTemperature
            self.testResult[month]['intRemain'] = intLoadRemain
            self.testResult[month]['unloadRemain'] = unLoadRemain
            self.testResult[month]['intUserPreference'] = intUserPreference
            self.testResult[month]['unintUserPreference'] = unintUserPreference
            self.testResult[month]['intSwitch'] = intSwitch
            self.testResult[month]['unintSwitch'] = unintSwitch
            self.testResult[month]['reward'] = Reward
            # print("intLoadRemain",intLoadRemain)
            # print("unLoadRemain",unLoadRemain)
            TotalReward.append(totalReward)
            totalReward=0
            sampletime.clear()
            remain.clear()
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
            intUserPreference.clear()
            unintUserPreference.clear()
            intSwitch.clear()
            unintSwitch.clear()
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
        output.price()
        output.plotIntPreference()
        output.plotUnintPreference()
        output.plotIntLoadPower()
        output.plotUnIntLoadPower()
        output.plotDeltaSOCPower()
        output.plotReward()
        output.plotResult('lib/plot/HRL/')

    def getMean(self):
        return super().getMean()

    def getStd(self):
        return super().getStd()
        
    def __del__(self):
        return super().__del__()
