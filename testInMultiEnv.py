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


            remain = [load[sampletime]+hvac[sampletime]+socPower[sampletime]-pv[sampletime] for sampletime in range(95)]
            self.price = [(self.price[month]-np.min(self.price))/(np.max(self.price)-np.min(self.price)) for month in range(len(self.price))]  

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

    def __plotResult__(self):
        plt.rcParams["figure.figsize"] = (12.8, 9.6)
        fig,axes = plt.subplots(6,2)
        plt.title("SOC and Price for each month")
        ax1=axes[0,0]
        ax2=axes[0,1]
        ax3=axes[1,0]
        ax4=axes[1,1]
        ax5=axes[2,0]
        ax6=axes[2,1]
        ax7=axes[3,0]
        ax8=axes[3,1]
        ax9=axes[4,0]
        ax10=axes[4,1]
        ax11=axes[5,0]
        ax12=axes[5,1]
        sub1=ax1.twinx()
        sub2=ax2.twinx()
        sub3=ax3.twinx()
        sub4=ax4.twinx()
        sub5=ax5.twinx()
        sub6=ax6.twinx()
        sub7=ax7.twinx()
        sub8=ax8.twinx()
        sub9=ax9.twinx()
        sub10=ax10.twinx()
        sub11=ax11.twinx()
        sub12=ax12.twinx()

        ax1.set_ylabel('SOC')
        ax1.plot(range(len(self.monthlySoc['1'][:])), self.monthlySoc['1'][:], label = "Jan",color='dimgray')    
        ax1.plot(range(len(self.price)), self.price, label = "price")
        ax1.set_title('Jan')

        ax2.set_ylabel('SOC')
        ax2.plot(range(len(self.monthlySoc['2'][:])), self.monthlySoc['2'][:], label = "Feb",color='dimgray')
        ax2.plot(range(len(self.price)), self.price, label = "price")
        ax2.set_title('Feb')

        ax3.set_ylabel('SOC')
        ax3.plot(range(len(self.price)), self.price, label = "price")
        ax3.plot(range(len(self.monthlySoc['3'][:])), self.monthlySoc['3'][:], label = "Mar",color='dimgray')
        ax3.set_title('Mar')

        ax4.set_ylabel('SOC')
        ax4.plot(range(len(self.monthlySoc['4'][:])), self.monthlySoc['4'][:], label = "Apr",color='dimgray')
        ax4.plot(range(len(self.price)), self.price, label = "price")
        ax4.set_title('Apr')

        ax5.set_ylabel('SOC')
        ax5.plot(range(len(self.monthlySoc['5'][:])), self.monthlySoc['5'][:], label = "May",color='dimgray')
        ax5.plot(range(len(self.price)), self.price, label = "price")
        ax5.set_title('May')

        ax6.set_ylabel('SOC')
        ax6.plot(range(len(self.monthlySoc['6'][:])), self.monthlySoc['6'][:], label = "Jun",color='dimgray')
        ax6.plot(range(len(self.price)), self.price, label = "price")
        ax6.set_title('Jun')

        ax7.set_ylabel('SOC')
        ax7.plot(range(len(self.monthlySoc['7'][:])), self.monthlySoc['7'][:], label = "July",color='dimgray')
        ax7.plot(range(len(self.price)), self.price, label = "price")
        ax7.set_title('July')

        ax8.set_ylabel('SOC')
        ax8.plot(range(len(self.monthlySoc['8'][:])), self.monthlySoc['8'][:], label = "Aug",color='dimgray')
        ax8.plot(range(len(self.price)), self.price, label = "price")
        ax8.set_title('Aug')

        ax9.set_ylabel('SOC')
        ax9.plot(range(len(self.monthlySoc['9'][:])), self.monthlySoc['9'][:], label = "Sep",color='dimgray')
        ax9.plot(range(len(self.price)), self.price, label = "price")
        ax9.set_title('Sep')

        ax10.set_ylabel('SOC')
        ax10.plot(range(len(self.monthlySoc['10'][:])), self.monthlySoc['10'][:], label = "Oct",color='dimgray')
        ax10.plot(range(len(self.price)), self.price, label = "price")
        ax10.set_title('Oct')

        ax11.set_ylabel('SOC')
        ax11.plot(range(len(self.monthlySoc['11'][:])), self.monthlySoc['11'][:], label = "Nov",color='dimgray')
        ax11.plot(range(len(self.price)), self.price, label = "price")
        ax11.set_title('Nov')

        ax12.set_ylabel('SOC')
        ax12.plot(range(len(self.monthlySoc['12'][:])), self.monthlySoc['12'][:], label = "Dec",color='dimgray')
        ax12.plot(range(len(self.price)), self.price, label = "price")
        ax12.set_title('Dec')

#-----------------------------------------------------------------------------------------------#

        sub1.set_ylabel('Power')
        sub1.bar(np.arange(95) ,self.monthlyRemain['1'][:] ,label = 'fixLoad',bottom = self.monthlySocPower['1'][:] +self.monthlyHVAC['1'][:], color ='gainsboro')  
        sub1.bar(np.arange(95) ,self.monthlySocPower['1'][:] ,label = 'socPower',bottom=self.monthlyHVAC['1'][:],color ='dimgray')  
        sub1.bar(np.arange(95) ,self.monthlyHVAC['1'][:] ,label = 'HVAC',color ='red')  

        sub2.set_ylabel('Power')
        sub2.bar(np.arange(95) ,self.monthlyRemain['2'][:] ,label = 'fixLoad',bottom = self.monthlySocPower['2'][:] +self.monthlyHVAC['2'][:], color ='gainsboro')  
        sub2.bar(np.arange(95) ,self.monthlySocPower['2'][:] ,label = 'socPower',bottom=self.monthlyHVAC['2'][:],color ='dimgray')  
        sub2.bar(np.arange(95) ,self.monthlyHVAC['2'][:] ,label = 'HVAC',color ='red')  

        sub3.set_ylabel('Power')
        sub3.bar(np.arange(95) ,self.monthlyRemain['3'][:] ,label = 'fixLoad',bottom = self.monthlySocPower['3'][:] +self.monthlyHVAC['3'][:], color ='gainsboro')  
        sub3.bar(np.arange(95) ,self.monthlySocPower['3'][:] ,label = 'socPower',bottom=self.monthlyHVAC['3'][:],color ='dimgray')  
        sub3.bar(np.arange(95) ,self.monthlyHVAC['3'][:] ,label = 'HVAC',color ='red')  

        sub4.set_ylabel('Power')
        sub4.bar(np.arange(95) ,self.monthlyRemain['4'][:] ,label = 'fixLoad',bottom = self.monthlySocPower['4'][:] +self.monthlyHVAC['4'][:], color ='gainsboro')  
        sub4.bar(np.arange(95) ,self.monthlySocPower['4'][:] ,label = 'socPower',bottom=self.monthlyHVAC['4'][:],color ='dimgray')  
        sub4.bar(np.arange(95) ,self.monthlyHVAC['4'][:] ,label = 'HVAC',color ='red')  

        sub5.set_ylabel('Power')
        sub5.bar(np.arange(95) ,self.monthlyRemain['5'][:] ,label = 'fixLoad',bottom = self.monthlySocPower['5'][:] +self.monthlyHVAC['5'][:], color ='gainsboro')  
        sub5.bar(np.arange(95) ,self.monthlySocPower['5'][:] ,label = 'socPower',bottom=self.monthlyHVAC['5'][:],color ='dimgray')  
        sub5.bar(np.arange(95) ,self.monthlyHVAC['5'][:] ,label = 'HVAC',color ='red')  

        sub6.set_ylabel('Power')
        sub6.bar(np.arange(95) ,self.monthlyRemain['6'][:] ,label = 'fixLoad',bottom = self.monthlySocPower['6'][:] +self.monthlyHVAC['6'][:], color ='gainsboro')  
        sub6.bar(np.arange(95) ,self.monthlySocPower['6'][:] ,label = 'socPower',bottom=self.monthlyHVAC['6'][:],color ='dimgray')  
        sub6.bar(np.arange(95) ,self.monthlyHVAC['6'][:] ,label = 'HVAC',color ='red')  

        sub7.set_ylabel('Power')
        sub7.bar(np.arange(95) ,self.monthlyRemain['7'][:] ,label = 'fixLoad',bottom = self.monthlySocPower['7'][:] +self.monthlyHVAC['7'][:], color ='gainsboro')  
        sub7.bar(np.arange(95) ,self.monthlySocPower['7'][:] ,label = 'socPower',bottom=self.monthlyHVAC['7'][:],color ='dimgray')  
        sub7.bar(np.arange(95) ,self.monthlyHVAC['7'][:] ,label = 'HVAC',color ='red')  

        sub8.set_ylabel('Power')
        sub8.bar(np.arange(95) ,self.monthlyRemain['8'][:] ,label = 'fixLoad',bottom = self.monthlySocPower['8'][:] +self.monthlyHVAC['8'][:], color ='gainsboro')  
        sub8.bar(np.arange(95) ,self.monthlySocPower['8'][:] ,label = 'socPower',bottom=self.monthlyHVAC['8'][:],color ='dimgray')  
        sub8.bar(np.arange(95) ,self.monthlyHVAC['8'][:] ,label = 'HVAC',color ='red')  

        sub9.set_ylabel('Power')
        sub9.bar(np.arange(95) ,self.monthlyRemain['9'][:] ,label = 'fixLoad',bottom = self.monthlySocPower['9'][:] +self.monthlyHVAC['9'][:], color ='gainsboro')  
        sub9.bar(np.arange(95) ,self.monthlySocPower['9'][:] ,label = 'socPower',bottom=self.monthlyHVAC['9'][:],color ='dimgray')  
        sub9.bar(np.arange(95) ,self.monthlyHVAC['9'][:] ,label = 'HVAC',color ='red')  

        sub10.set_ylabel('Power')
        sub10.bar(np.arange(95) ,self.monthlyRemain['10'][:] ,label = 'fixLoad',bottom = self.monthlySocPower['10'][:]+self.monthlyHVAC['10'][:] , color ='gainsboro')  
        sub10.bar(np.arange(95) ,self.monthlySocPower['10'][:] ,label = 'socPower',bottom=self.monthlyHVAC['10'][:], color ='red')  
        sub10.bar(np.arange(95) ,self.monthlyHVAC['10'][:] ,label = 'HVAC', color ='red')  

        sub11.set_ylabel('Power')
        sub11.bar(np.arange(95) ,self.monthlyRemain['11'][:] ,label = 'fixLoad',bottom = self.monthlySocPower['11'][:]+self.monthlyHVAC['11'][:] , color ='gainsboro')  
        sub11.bar(np.arange(95) ,self.monthlySocPower['11'][:] ,label = 'socPower',bottom=self.monthlyHVAC['11'][:],color ='dimgray')  
        sub11.bar(np.arange(95) ,self.monthlyHVAC['11'][:] ,label = 'HVAC', color ='red')  

        sub12.set_ylabel('Power')
        sub12.bar(np.arange(95) ,self.monthlyRemain['12'][:] ,label = 'fixLoad',bottom = self.monthlySocPower['12'][:]+self.monthlyHVAC['12'][:] , color ='gainsboro')  
        sub12.bar(np.arange(95) ,self.monthlySocPower['12'][:] ,label = 'socPower',bottom=self.monthlyHVAC['12'][:],color ='dimgray')
        sub12.bar(np.arange(95) ,self.monthlyHVAC['12'][:] ,label = 'HVAC', color ='red')  

        fig.tight_layout()
        fig.savefig('pic/multi/Multiplot.png')

    def __del__(self):
        self.hvacEnv.close()
        self.socEnv.close()
        self.hvacAgent.close()
        self.socAgent.close()

if __name__ == '__main__':
    env = MultiEnvTest()
    env.__testInHvacSoc__()
    env.__plotResult__()