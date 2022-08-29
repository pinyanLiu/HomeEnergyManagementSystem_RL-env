from gym import make
from tensorforce import Agent,Environment
#from lib.loads.interrupted import AC
#from lib.loads.uninterrupted import WM
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
class MultiEnvTest():
    def __init__(self):
        pass
    def __testInHvacSoc__(self):
        self.hvacSocEnv = make("Multi-hems-v0")
#     # Initialize episode
        self.hvacEnv = Environment.create(environment = 'gym',level='Hems-v7')
        self.socEnv = Environment.create(environment = 'gym',level='Hems-v1')
        self.hvacAgent = Agent.load(directory = 'HVAC/saver_dir',environment=self.hvacEnv)
        self.socAgent = Agent.load(directory = 'Soc/saver_dir',environment=self.socEnv)
        load = []
        hvac = []  
        soc = []
        socPower = []
        pv = []
        indoorTemperature = []
        outdoorTemperature = []
        userSetTemperature = []
        totalReward = 0
        self.monthlySoc = pd.DataFrame()
        self.monthlySocPower = pd.DataFrame()
        self.monthlyIndoorTemperature = pd.DataFrame()
        self.monthlyOutdoorTemperature = pd.DataFrame()
        self.monthlyRemain = pd.DataFrame()
        self.monthlyHVAC = pd.DataFrame()
        self.monthlyUserSetTemperature = pd.DataFrame()
        self.price = []

        for month in range(12):

            states = self.hvacSocEnv.reset()
            self.hvacEnv.reset()
            self.socEnv.reset()
            hvacInternals = self.hvacAgent.initial_internals()
            socInternals = self.socAgent.initial_internals()
            terminal = False
            while not terminal:
                print(states)
            #get hvac state
                hvacStates = []
                socStates = []
                hvacStates.extend(states[:3])
                hvacStates.extend(states[4:])#exclude soc
            #hvac act
                hvacActions, hvacInternals = self.hvacAgent.act(
                    states=hvacStates, internals=hvacInternals, independent=True, deterministic=True
                )
                hvacStates, hvacTerminal, hvacReward = self.hvacEnv.execute(actions=hvacActions)
            # update hvacSocEnv State
                states[1] += hvacActions # fixload += power of hvac
                states[5] = hvacStates[4] #indoor temperature
            #get soc state
                socStates = states[:5]
                socActions, socInternals = self.socAgent.act(
                    states=socStates, internals=socInternals, independent=True, deterministic=True
                )
                socStates, socTerminal, socReward = self.socEnv.execute(actions=socActions)
            #update hvacSocEnv state
                states[3] = socStates[3]



                load.append(hvacStates[1])
                pv.append(states[2])
                soc.append(states[3])
                if month == 11:
                    self.price.append(states[4])
                indoorTemperature.append(states[5])
                outdoorTemperature.append(states[6])
                userSetTemperature.append(states[7])
                hvac.append(hvacActions[0])
                socPower.append(socActions[0]*3)
                totalReward  = hvacReward+socReward
                #hvacSocEnv step()
                actions = self.hvacSocEnv.action_space.sample()
                states, reward, terminal , info = self.hvacSocEnv.step(action=actions)


            remain = [load[sampletime]-pv[sampletime] for sampletime in range(95)]
            #store testing result in each dictionary
            self.monthlySoc.insert(month,column=str(month+1),value=soc)
            self.monthlySocPower.insert(month,column=str(month+1),value=socPower)
            self.monthlyIndoorTemperature.insert(month,column=str(month+1),value=indoorTemperature)
            self.monthlyOutdoorTemperature.insert(month,column=str(month+1),value=outdoorTemperature)
            self.monthlyRemain.insert(month,column=str(month+1),value=remain)
            self.monthlyHVAC.insert(month,column=str(month+1),value=hvac)
            self.monthlyUserSetTemperature.insert(month,column=str(month+1),value=userSetTemperature)

            load.clear()
            pv.clear()
            soc.clear()
            socPower.clear()
            indoorTemperature.clear()
            outdoorTemperature.clear()
            userSetTemperature.clear()
            hvac.clear()
        print('Agent average episode reward: ', totalReward/12 )
    def __del__(self):
        self.hvacEnv.close()
        self.socEnv.close()
        self.hvacAgent.close()
        self.socAgent.close()

if __name__ == '__main__':
    env = MultiEnvTest()
    env.__testInHvacSoc__()