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
        TotalHvacPreference = []
        hvacPower = []
        intLoadRemain = []
        unLoadRemain = []
        intUserPreference = []
        unintUserPreference = []
        intSwitch = []
        unintSwitch = []
        order = []
        Reward = []
        TotalReward = []
        TotalElectricPrice = []
        TotalIntPreference = []
        TotalUnintPreference = []
        ExceedPgridMaxTimes = []
        PgridMax = []
        totalReward = 0
        for month in range(12):
            states = self.environment.reset()
            totalState = self.environment.totalState
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
                    
                    TotalHvacPreference.append(states['state'][4])
                    hvacPower.append(totalState['hvacPower'])
                    intLoadRemain.append(totalState["intRemain"])
                    unLoadRemain.append(totalState["unintRemain"])
                    intUserPreference.append(totalState["intPreference"])
                    unintUserPreference.append(totalState["unintPreference"])
                    intSwitch.append(totalState["intSwitch"])
                    unintSwitch.append(totalState["unintSwitch"])
                    order.append(totalState["order"])
                    Reward.append(reward)
                    TotalElectricPrice.append(0.25*totalState["pricePerHour"]*states['state'][2] if states['state'][2]>0 else 0)
                    TotalIntPreference.append(totalState["intSwitch"]*totalState["intPreference"])
                    TotalUnintPreference.append(totalState["unintSwitch"]*totalState["unintPreference"])
                    if states['state'][2]>totalState['PgridMax']:
                        ExceedPgridMaxTimes.append(1)
                    else:
                        ExceedPgridMaxTimes.append(0)
                    # ExceedPgridMaxTimes.append(1 if states['state'][2]>totalState['PgridMax'] else 0)
                    PgridMax.append(totalState['PgridMax'])
                totalReward += reward
            self.testResult[month]['sampleTime'] = sampletime
            self.testResult[month]['remain'] = remain
            self.testResult[month]['price'] = price
            self.testResult[month]['soc'] = soc
            self.testResult[month]['PV'] = pv
            self.testResult[month]['deltaSoc'] = deltaSoc
            self.testResult[month]['indoorTemperature'] = indoorTemperature
            self.testResult[month]['outdoorTemperature'] = outdoorTemperature
            self.testResult[month]['userSetTemperature'] = userSetTemperature
            self.testResult[month]['TotalHvacPreference'] = TotalHvacPreference
            self.testResult[month]['hvacPower'] = hvacPower
            self.testResult[month]['intRemain'] = intLoadRemain
            self.testResult[month]['unloadRemain'] = unLoadRemain
            self.testResult[month]['intUserPreference'] = intUserPreference
            self.testResult[month]['unintUserPreference'] = unintUserPreference
            self.testResult[month]['intSwitch'] = intSwitch
            self.testResult[month]['unintSwitch'] = unintSwitch
            self.testResult[month]['reward'] = Reward
            self.testResult[month]["TotalElectricPrice"] = TotalElectricPrice
            self.testResult[month]["TotalIntPreference"] = TotalIntPreference
            self.testResult[month]["TotalUnintPreference"] = TotalUnintPreference
            self.testResult[month]["ExceedPgridMaxTimes"] = ExceedPgridMaxTimes
            self.testResult[month]["PgridMax"] = PgridMax
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
            TotalHvacPreference.clear()
            hvacPower.clear()
            intLoadRemain.clear()
            unLoadRemain.clear()
            intUserPreference.clear()
            unintUserPreference.clear()
            intSwitch.clear()
            unintSwitch.clear()
            order.clear()
            Reward.clear()
            TotalUnintPreference.clear()
            TotalIntPreference.clear()
            TotalElectricPrice.clear()
            ExceedPgridMaxTimes.clear()
            PgridMax.clear()

        print('Agent average episode reward: ', sum(TotalReward)/len(TotalReward) ) 
        print('reward: ', TotalReward ) 
        ExceedPgridMaxTimes = 0
        TotalIntPreference = 0 
        TotalUnintPreference = 0 
        TotalElectricPrice = 0
        TotalHvacPreference = 0
        for month in range(12):
            ExceedPgridMaxTimes += sum(self.testResult[month]["ExceedPgridMaxTimes"])
            TotalHvacPreference+= sum(self.testResult[month]["TotalHvacPreference"])
            TotalIntPreference  += sum(self.testResult[month]["TotalIntPreference"])
            TotalUnintPreference += sum(self.testResult[month]["TotalUnintPreference"])
            TotalElectricPrice += sum(self.testResult[month]["TotalElectricPrice"])
        print("Exceed PgridMax Ratio :", ExceedPgridMaxTimes/(12*96))
        print("AVG TotalHvacPreference :", TotalHvacPreference/12)
        print("AVG TotalIntPreference :", TotalIntPreference/12)
        print("AVG TotalUnintPreference:", TotalUnintPreference/12)
        print("AVG TotalElectricPrice:", TotalElectricPrice/12)

    
    def outputResult(self):
        output = Plot(self.testResult)
        output.remainPower()
        output.soc()
        output.indoorTemperature()
        output.outdoorTemperature()
        output.userSetTemperature()
        output.plotHVACPower()
        output.price()
        output.plotIntPreference()
        output.plotUnintPreference()
        output.plotIntLoadPower()
        output.plotUnIntLoadPower()
        output.plotDeltaSOCPower()
        output.plotPVPower()
        output.plotPgridMax()
        output.plotReward()
        output.plotResult('lib/plot/HRL/')

    def getMean(self):
        return super().getMean()

    def getStd(self):
        return super().getStd()
        
    def __del__(self):
        return super().__del__()
