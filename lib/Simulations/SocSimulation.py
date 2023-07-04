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
        ExceedPgridMaxTimes=[]
        totalReward = 0
        for month in range(12):
            states = self.environment.reset()
            sampletime.append(states[0])
            load.append(states[1])
            pv.append(states[2])
            soc.append(states[3])
            price.append(states[4])
            ExceedPgridMaxTimes.append(0)
            internals = self.agent.initial_internals()
            terminal = False
            while not terminal:
                actions, internals = self.agent.act(
                    states=states, internals=internals, independent=True, deterministic=True
                )
                states, terminal, reward = self.environment.execute(actions=actions)
                if actions == 0 :
                    action = 0.25
                elif actions ==1:
                    action = 0.2
                elif actions ==2:
                    action = 0.15
                elif actions ==3:
                    action = 0.1
                elif actions ==4:
                    action = 0.05
                elif actions ==5:
                    action = 0.00
                elif actions ==6:
                    action = -0.05
                elif actions ==7:
                    action = -0.1
                elif actions ==8:
                    action = -0.15
                elif actions ==9:
                    action = -0.2
                elif actions ==10:
                    action = -0.25
                deltaSoc.append(action)
                sampletime.append(states['state'][0])
                load.append(states['state'][1])
                pv.append(states['state'][2])
                soc.append(states['state'][3])
                price.append(states['state'][4])
                ExceedPgridMaxTimes.append(1 if action*10+states['state'][1]-states['state'][2]>10 else 0)
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
            self.testResult[month]['ExceedPgridMaxTimes']=ExceedPgridMaxTimes
            TotalReward.append(totalReward)
            totalReward=0
            sampletime.clear()
            load.clear()
            pv.clear()
            soc.clear()
            price.clear()
            deltaSoc.clear()
            ExceedPgridMaxTimes.clear()
            Reward.clear()
        for month in range(12):
            print("month ",month, " ExceedPgridMaxTimes: ",sum(self.testResult[month]["ExceedPgridMaxTimes"]))
        print('Agent average episode reward: ', sum(TotalReward)/len(TotalReward) ) 
        print('reward: ', TotalReward ) 
    
    def outputResult(self):
        # if month != False:
            output = Plot(self.testResult,single=False)
            output.remainPower(month=False)
            output.plotDeltaSOCPower(month=False)
            output.soc(month=False)
            output.price(month=False)
            output.plotResult('lib/plot/soc/')
        # else:
        #     output = Plot(self.testResult)
        #     output.remainPower()
        #     output.plotDeltaSOCPower()
        #     output.soc()
        #     output.price()
        #     output.plotReward()
        #     output.plotResult('lib/plot/soc/')

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


