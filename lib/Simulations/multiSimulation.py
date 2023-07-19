from lib.Simulations.Simulation import Simulation
from tensorforce import Agent,Environment
from lib.enviroment.multiAgentEnv.multiAgentTestEnv import multiAgentTestEnv
from lib.plot.plot import Plot


class multiSimulation(Simulation):
    def __init__(self):
        super().__init__()
        self.environment = Environment.create(environment = multiAgentTestEnv,max_episode_timesteps=960)
        self.agent = Agent.load(directory = 'HLA/saver_dir',environment=self.environment)

    def simulation(self):
        sampletime = []
        remain = []
        load = []
        pv = []
        soc = []
        price = []
        deltaSoc = []
        indoorTemperature1 = []
        indoorTemperature2 = []
        indoorTemperature3 = []
        outdoorTemperature = []
        userSetTemperature1 = []
        userSetTemperature2 = []
        userSetTemperature3 = []
        TotalHvacPreference = []
        hvacPower1 = []
        hvacPower2 = []
        hvacPower3 = []
        intLoadRemain1 = []
        intLoadRemain2 = []
        intLoadRemain3 = []
        unLoadRemain1 = []
        unLoadRemain2 = []
        intUserPreference1 = []
        intUserPreference2 = []
        intUserPreference3 = []
        unintUserPreference1 = []
        unintUserPreference2 = []
        intSwitch1 = []
        intSwitch2 = []
        intSwitch3 = []
        unintSwitch1 = []
        unintSwitch2= []
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
                if totalState["order"]==9:
                    sampletime.append(totalState["sampleTime"])
                    remain.append(states['state'][2])
                    load.append(totalState["fixLoad"])
                    pv.append(totalState["PV"])
                    soc.append(totalState["SOC"])
                    price.append(totalState["pricePerHour"])
                    deltaSoc.append(totalState["deltaSoc"])
                    indoorTemperature1.append(totalState["indoorTemperature1"])
                    indoorTemperature2.append(totalState["indoorTemperature2"])
                    indoorTemperature3.append(totalState["indoorTemperature3"])
                    outdoorTemperature.append(totalState["outdoorTemperature"])
                    userSetTemperature1.append(totalState["userSetTemperature1"])
                    userSetTemperature2.append(totalState["userSetTemperature2"])
                    userSetTemperature3.append(totalState["userSetTemperature3"])
                    TotalHvacPreference.append(states['state'][4]+states['state'][5]+states['state'][6])
                    hvacPower1.append(totalState['hvacPower1'])
                    hvacPower2.append(totalState['hvacPower2'])
                    hvacPower3.append(totalState['hvacPower3'])
                    intLoadRemain1.append(totalState["intRemain1"])
                    intLoadRemain2.append(totalState["intRemain2"])
                    intLoadRemain3.append(totalState["intRemain3"])
                    unLoadRemain1.append(totalState["unintRemain1"])
                    unLoadRemain2.append(totalState["unintRemain2"])
                    intUserPreference1.append(totalState["intPreference1"])
                    intUserPreference2.append(totalState["intPreference2"])
                    intUserPreference3.append(totalState["intPreference3"])
                    unintUserPreference1.append(totalState["unintPreference1"])
                    unintUserPreference2.append(totalState["unintPreference2"])
                    intSwitch1.append(totalState["intSwitch1"])
                    intSwitch2.append(totalState["intSwitch2"])
                    intSwitch3.append(totalState["intSwitch3"])
                    unintSwitch1.append(totalState["unintSwitch1"])
                    unintSwitch2.append(totalState["unintSwitch2"])
                    Reward.append(reward)
                    order.append(totalState["order"])
                    TotalElectricPrice.append(0.25*totalState["pricePerHour"]*states['state'][2] if states['state'][2]>0 else 0)
                    TotalIntPreference.append(totalState["intSwitch1"]*totalState["intPreference1"]+totalState["intSwitch2"]*totalState["intPreference2"]+totalState["intSwitch3"]*totalState["intPreference3"])
                    TotalUnintPreference.append(totalState["unintSwitch1"]*totalState["unintPreference1"]+totalState["unintSwitch2"]*totalState["unintPreference2"])              
                    ExceedPgridMaxTimes.append(1 if states['state'][2]>totalState['PgridMax'] else 0)
                    PgridMax.append(totalState['PgridMax'])
                totalReward += reward
            self.testResult[month]['sampleTime'] = sampletime
            self.testResult[month]['remain'] = remain
            self.testResult[month]['load'] = load
            self.testResult[month]['price'] = price
            self.testResult[month]['soc'] = soc
            self.testResult[month]['PV'] = pv
            self.testResult[month]['deltaSoc'] = deltaSoc
            self.testResult[month]['indoorTemperature1'] = indoorTemperature1
            self.testResult[month]['indoorTemperature2'] = indoorTemperature2
            self.testResult[month]['indoorTemperature3'] = indoorTemperature3
            self.testResult[month]['outdoorTemperature'] = outdoorTemperature
            self.testResult[month]['userSetTemperature1'] = userSetTemperature1
            self.testResult[month]['userSetTemperature2'] = userSetTemperature2
            self.testResult[month]['userSetTemperature3'] = userSetTemperature3
            self.testResult[month]['TotalHvacPreference'] = TotalHvacPreference
            self.testResult[month]['hvacPower1'] = hvacPower1
            self.testResult[month]['hvacPower2'] = hvacPower2
            self.testResult[month]['hvacPower3'] = hvacPower3
            self.testResult[month]['intRemain1'] = intLoadRemain1
            self.testResult[month]['intRemain2'] = intLoadRemain2
            self.testResult[month]['intRemain3'] = intLoadRemain3
            self.testResult[month]['unloadRemain1'] = unLoadRemain1
            self.testResult[month]['unloadRemain2'] = unLoadRemain2
            self.testResult[month]['intUserPreference1'] = intUserPreference1
            self.testResult[month]['intUserPreference2'] = intUserPreference2
            self.testResult[month]['intUserPreference3'] = intUserPreference3
            self.testResult[month]['unintUserPreference1'] = unintUserPreference1
            self.testResult[month]['unintUserPreference2'] = unintUserPreference2
            self.testResult[month]['intSwitch1'] = intSwitch1
            self.testResult[month]['intSwitch2'] = intSwitch2
            self.testResult[month]['intSwitch3'] = intSwitch3
            self.testResult[month]['unintSwitch1'] = unintSwitch1
            self.testResult[month]['unintSwitch2'] = unintSwitch2
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
            indoorTemperature1.clear()
            indoorTemperature2.clear()
            indoorTemperature3.clear()
            outdoorTemperature.clear()
            userSetTemperature1.clear()
            userSetTemperature2.clear()
            userSetTemperature3.clear()
            TotalHvacPreference.clear()
            hvacPower1.clear()
            hvacPower2.clear()
            hvacPower3.clear()
            intLoadRemain1.clear()
            intLoadRemain2.clear()
            intLoadRemain3.clear()
            unLoadRemain1.clear()
            unLoadRemain2.clear()
            intUserPreference1.clear()
            intUserPreference2.clear()
            intUserPreference3.clear()
            unintUserPreference1.clear()
            unintUserPreference2.clear()
            intSwitch1.clear()
            intSwitch2.clear()
            intSwitch3.clear()
            unintSwitch1.clear()
            unintSwitch2.clear()
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

    def EachMonthResult(self):
        return super().EachMonthResult()

    
    def outputResult(self,mode):
        output = Plot(self.testResult)
        if mode == "hvac":
            output.fixloadPower()
            output.soc()
            output.indoorTemperature(1)
            output.indoorTemperature(2)
            output.indoorTemperature(3)
            output.outdoorTemperature()
            output.userSetTemperature(1)
            output.userSetTemperature(2)
            output.userSetTemperature(3)
            output.plotHVACPower(id=1)
            output.plotHVACPower(id=2)
            output.plotHVACPower(id=3)
            output.plotDeltaSOCPower()
            output.price()
            output.plotPgridMax()
            output.plotPVPower()
            output.plotResult('lib/plot/HRL/hvac/')

        elif mode == "soc":
            output.fixloadPower()
            output.soc()
            output.price()
            output.plotPgridMax()
            output.plotPVPower()
            output.plotDeltaSOCPower()
            output.plotResult('lib/plot/HRL/soc/')
            
        elif mode == "int":
            output.fixloadPower()
            output.soc()
            output.price()
            output.plotPgridMax()
            output.plotPVPower()
            output.plotIntPreference(1)
            output.plotIntPreference(2)
            output.plotIntPreference(3)
            output.plotIntLoadPower(1)
            output.plotIntLoadPower(2)
            output.plotIntLoadPower(3)
            output.plotDeltaSOCPower()
            output.plotResult('lib/plot/HRL/int/')

        elif mode == "unint":
            output.fixloadPower()
            output.soc()
            output.price()
            output.plotPgridMax()
            output.plotPVPower()
            output.plotUnintPreference(1)
            output.plotUnintPreference(2)
            output.plotUnIntLoadPower(1)
            output.plotUnIntLoadPower(2)
            output.plotResult('lib/plot/HRL/unint/')
        else:
            output.fixloadPower()
            output.soc()
            output.price()
            output.plotPgridMax()
            output.plotDeltaSOCPower()
            output.plotPVPower()
            output.plotHVACPower(id=1)
            output.plotHVACPower(id=2)
            output.plotHVACPower(id=3)
            output.plotIntPreference(1)
            output.plotIntPreference(2)
            output.plotIntPreference(3)
            output.plotIntLoadPower(1)
            output.plotIntLoadPower(2)
            output.plotIntLoadPower(3)
            output.plotUnintPreference(1)
            output.plotUnintPreference(2)
            output.plotUnIntLoadPower(1)
            output.plotUnIntLoadPower(2)
            output.plotResult('lib/plot/HRL/')



    def getMean(self):
        return super().getMean()

    def getStd(self):
        return super().getStd()
        
    def __del__(self):
        return super().__del__()
