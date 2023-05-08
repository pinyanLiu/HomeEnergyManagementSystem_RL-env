from lib.Simulations.multiSimulation import multiSimulation
from tensorforce import Agent,Environment
from lib.Simulations.Simulation import Simulation
from lib.plot.plot import Plot

from lib.enviroment.multiAgentEnv.OneLevelTestEnv import OneLevelTestEnv
import pandas as pd
class OneLevelSimulation(multiSimulation):
    def __init__(self):
        self.testResult = {}
        for month in range(12):
            self.testResult[month] = pd.DataFrame()
        self.totalReward = []        
        self.environment = Environment.create(environment = OneLevelTestEnv,max_episode_timesteps=672)
        self.agent = Agent.load(directory = 'HLA/OneLevelAgent/saver_dir',environment=self.environment)

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
                if totalState["order"]==1:
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
                    ExceedPgridMaxTimes.append(1 if states['state'][2]>totalState['PgridMax'] else 0)

                    # ExceedPgridMaxTimes.append(1 if states['state'][2]>totalState['PgridMax'] else 0)
                    PgridMax.append(totalState['PgridMax'])
                totalReward += reward
            self.testResult[month]['sampleTime'] = sampletime
            self.testResult[month]['remain'] = remain
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
            TotalHvacPreference  += sum(self.testResult[month]["TotalHvacPreference"])
            TotalIntPreference  += sum(self.testResult[month]["TotalIntPreference"])
            TotalUnintPreference += sum(self.testResult[month]["TotalUnintPreference"])
            TotalElectricPrice += sum(self.testResult[month]["TotalElectricPrice"])
        print("Exceed PgridMax Ratio :", ExceedPgridMaxTimes/(12*96)*100,"%")
        print("AVG TotalHvacPreference :", TotalHvacPreference/12)
        print("AVG TotalIntPreference :", TotalIntPreference/12)
        print("AVG TotalUnintPreference:", TotalUnintPreference/12)
        print("AVG TotalElectricPrice:", TotalElectricPrice/12)    

    def EachMonthResult(self):
        for month in range(12):
            print("month ",month, " ExceedPgridMaxTimes: ",sum(self.testResult[month]["ExceedPgridMaxTimes"]))
            print("month ",month, " TotalHvacPreference: ",sum(self.testResult[month]["TotalHvacPreference"]))
            print("month ",month, " TotalIntPreference: ",sum(self.testResult[month]["TotalIntPreference"]))
            print("month ",month, " TotalUnintPreference: ",sum(self.testResult[month]["TotalUnintPreference"]))
            print("month ",month, " TotalElectricPrice: ",sum(self.testResult[month]["TotalElectricPrice"]))

    def outputResult(self):
        output = Plot(self.testResult)
        output.remainPower()
        output.soc()
        output.indoorTemperature(1)
        output.indoorTemperature(2)
        output.indoorTemperature(3)
        output.outdoorTemperature()
        output.userSetTemperature(1)
        output.userSetTemperature(2)
        output.userSetTemperature(3)
        output.price()
        output.plotIntPreference()
        output.plotUnintPreference()
        output.plotIntLoadPower()
        output.plotUnIntLoadPower()
        output.plotDeltaSOCPower()
        output.plotHVACPower(id=1)
        output.plotHVACPower(id=2)
        output.plotHVACPower(id=3)
        output.plotPVPower()
        output.plotPgridMax()
        #output.plotReward()
        output.plotResult('lib/plot/OneLevel/')    
    def __del__(self):
        return super().__del__()
    
