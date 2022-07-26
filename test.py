from tensorforce import Agent,Environment
from lib.loads.interrupted import AC
from lib.loads.uninterrupted import WM
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

class Test():
    def __init__(self,mode) :
        self.mode = mode

    def main(self):
    #run test result in different env
        if self.mode == 'soc':
            self.__testInSoc__()
        
        elif self.mode == 'load':
            self.__testInLoad__()
    #plot result
        self.__plotResult__()

    def __testInSoc__(self):
        self.environment = Environment.create(environment='gym',level='Hems-v1')
        self.agent = Agent.load(directory = 'Soc/saver_dir',environment=self.environment)
        soc = []
        load = []
        pv = []
        totalReward = 0
        self.monthlySoc = pd.DataFrame()
        self.monthlyRemain = pd.DataFrame()
        self.price = []
        for month in range(12):
            states = self.environment.reset()
            internals = self.agent.initial_internals()
            terminal = False
            while not terminal:
                actions, internals = self.agent.act(
                    states=states, internals=internals, independent=True, deterministic=True
                )
                states, terminal, reward = self.environment.execute(actions=actions)
                load.append(states[1])
                pv.append(states[2])
                soc.append(states[3])
                totalReward += reward
                if month == 11:
                    self.price.append(states[4])

            remain = [load[sampletime]-pv[sampletime] for sampletime in range(95)]
            #normalize price to [0,1]
            self.price = [(self.price[month]-np.min(self.price))/(np.max(self.price)-np.min(self.price)) for month in range(len(self.price))]  
            self.monthlySoc.insert(month,column=str(month+1),value=soc)
            # monthlyLoad.insert(month,column=str(month+1),value=load)
            # monthlyPv.insert(month,column=str(month+1),value=pv)
            self.monthlyRemain.insert(month,column=str(month+1),value=remain)
            load.clear()
            pv.clear()
            soc.clear()
        print('Agent average episode reward: ', totalReward/12 )

    def __testInLoad__(self):
        self.environment = Environment.create(environment='gym',level='Hems-v5')
        self.agent = Agent.load(directory = 'Load/saver_dir',environment=self.environment)
        ac_object = AC(demand=40,AvgPowerConsume=3000)
        wm_object = WM(demand=40,AvgPowerConsume=3000,executePeriod=40)
        load = []
        pv = []
        ac = []
        wm = []
        totalReward = 0
        self.monthlyRemain = pd.DataFrame()
        self.acConsume = pd.DataFrame()
        self.wmConsume = pd.DataFrame()
        self.price = []
        for month in range(12):
            states = self.environment.reset()
            internals = self.agent.initial_internals()
            terminal = False
            while not terminal:
                actions, internals = self.agent.act(
                    states=states, internals=internals, independent=True, deterministic=True
                )
                states, terminal, reward = self.environment.execute(actions=actions)
                #1. AC on , WM on
                if actions == 0:
                    ac_object.turn_on()
                    wm_object.turn_on()
                    ac.append(3000)#power
                    wm.append(3000)
                #2. AC on , WM off
                elif actions == 1:
                    ac_object.turn_on()
                    ac.append(3000)
                    if(wm_object.reachExecutePeriod() == False):
                        wm_object.turn_on()
                        wm.append(3000)
                    else:
                        wm_object.turn_off
                        wm.append(0)
                # AC off , WM on
                elif actions == 2 :
                    ac_object.turn_off()
                    wm_object.turn_on()
                    ac.append(0)
                    wm.append(3000)
                # AC off , WM off
                else:
                    ac_object.turn_off()
                    ac.append(0)
                    if(wm_object.reachExecutePeriod() == False):
                        wm_object.turn_on()
                        wm.append(3000)
                    else:
                        wm_object.turn_off
                        wm.append(0)
                load.append(states[1])
                pv.append(states[2])
                totalReward += reward
                if month == 11:
                    self.price.append(states[3])

            remain = [load[sampletime]-pv[sampletime] for sampletime in range(95)]
            #normalize price to [0,1]
            self.price = [(self.price[month]-np.min(self.price))/(np.max(self.price)-np.min(self.price)) for month in range(len(self.price))]  
            self.monthlyRemain.insert(month,column=str(month+1),value=remain)
            self.acConsume.insert(month,column=str(month+1),value=ac)
            self.wmConsume.insert(month,column=str(month+1),value=wm)
            ac.clear()
            wm.clear()
            load.clear()
            pv.clear()
        print('Agent average episode reward: ', totalReward/12 )

    def __plotResult__(self):
        plt.rcParams["figure.figsize"] = (12.8, 9.6)
        fig,axes = plt.subplots(6,2)
        plt.title("SOC and Price for each month")
        #plt.xlabel("sampleTime")
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

        if self.mode == 'soc':
            ax1.set_ylabel('SOC')
            ax1.plot(range(len(self.monthlySoc['1'][:])), self.monthlySoc['1'][:], label = "Jan",color='red')    
            ax1.plot(range(len(self.price)), self.price, label = "price")
            ax1.set_title('Jan')

            ax2.set_ylabel('SOC')
            ax2.plot(range(len(self.monthlySoc['2'][:])), self.monthlySoc['2'][:], label = "Feb",color='red')
            ax2.plot(range(len(self.price)), self.price, label = "price")
            ax2.set_title('Feb')

            ax3.set_ylabel('SOC')
            ax3.plot(range(len(self.price)), self.price, label = "price")
            ax3.plot(range(len(self.monthlySoc['3'][:])), self.monthlySoc['3'][:], label = "Mar",color='red')
            ax3.set_title('Mar')

            ax4.set_ylabel('SOC')
            ax4.plot(range(len(self.monthlySoc['4'][:])), self.monthlySoc['4'][:], label = "Apr",color='red')
            ax4.plot(range(len(self.price)), self.price, label = "price")
            ax4.set_title('Apr')

            ax5.set_ylabel('SOC')
            ax5.plot(range(len(self.monthlySoc['5'][:])), self.monthlySoc['5'][:], label = "May",color='red')
            ax5.plot(range(len(self.price)), self.price, label = "price")
            ax5.set_title('May')

            ax6.set_ylabel('SOC')
            ax6.plot(range(len(self.monthlySoc['6'][:])), self.monthlySoc['6'][:], label = "Jun",color='red')
            ax6.plot(range(len(self.price)), self.price, label = "price")
            ax6.set_title('Jun')

            ax7.set_ylabel('SOC')
            ax7.plot(range(len(self.monthlySoc['7'][:])), self.monthlySoc['7'][:], label = "July",color='red')
            ax7.plot(range(len(self.price)), self.price, label = "price")
            ax7.set_title('July')

            ax8.set_ylabel('SOC')
            ax8.plot(range(len(self.monthlySoc['8'][:])), self.monthlySoc['8'][:], label = "Aug",color='red')
            ax8.plot(range(len(self.price)), self.price, label = "price")
            ax8.set_title('Aug')

            ax9.set_ylabel('SOC')
            ax9.plot(range(len(self.monthlySoc['9'][:])), self.monthlySoc['9'][:], label = "Sep",color='red')
            ax9.plot(range(len(self.price)), self.price, label = "price")
            ax9.set_title('Sep')

            ax10.set_ylabel('SOC')
            ax10.plot(range(len(self.monthlySoc['10'][:])), self.monthlySoc['10'][:], label = "Oct",color='red')
            ax10.plot(range(len(self.price)), self.price, label = "price")
            ax10.set_title('Oct')

            ax11.set_ylabel('SOC')
            ax11.plot(range(len(self.monthlySoc['11'][:])), self.monthlySoc['11'][:], label = "Nov",color='red')
            ax11.plot(range(len(self.price)), self.price, label = "price")
            ax11.set_title('Nov')

            ax12.set_ylabel('SOC')
            ax12.plot(range(len(self.monthlySoc['12'][:])), self.monthlySoc['12'][:], label = "Dec",color='red')
            ax12.plot(range(len(self.price)), self.price, label = "price")
            ax12.set_title('Dec')

            sub1.set_ylabel('Power')
            sub1.plot(range(len(self.monthlyRemain['1'][:])), self.monthlyRemain['1'][:],color='green')  

            sub2.set_ylabel('Power')
            sub2.plot(range(len(self.monthlyRemain['2'][:])), self.monthlyRemain['2'][:],color='green')  

            sub3.set_ylabel('Power')
            sub3.plot(range(len(self.monthlyRemain['3'][:])), self.monthlyRemain['3'][:],color='green')  

            sub4.set_ylabel('Power')
            sub4.plot(range(len(self.monthlyRemain['4'][:])), self.monthlyRemain['4'][:],color='green')  

            sub5.set_ylabel('Power')
            sub5.plot(range(len(self.monthlyRemain['5'][:])), self.monthlyRemain['5'][:],color='green')  

            sub6.set_ylabel('Power')
            sub6.plot(range(len(self.monthlyRemain['6'][:])), self.monthlyRemain['6'][:],color='green')  

            sub7.set_ylabel('Power')
            sub7.plot(range(len(self.monthlyRemain['7'][:])), self.monthlyRemain['7'][:],color='green')  

            sub8.set_ylabel('Power')
            sub8.plot(range(len(self.monthlyRemain['8'][:])), self.monthlyRemain['8'][:],color='green')  

            sub9.set_ylabel('Power')
            sub9.plot(range(len(self.monthlyRemain['9'][:])), self.monthlyRemain['9'][:],color='green')  

            sub10.set_ylabel('Power')
            sub10.plot(range(len(self.monthlyRemain['10'][:])), self.monthlyRemain['10'][:],color='green')  

            sub11.set_ylabel('Power')
            sub11.plot(range(len(self.monthlyRemain['11'][:])), self.monthlyRemain['11'][:],color='green')  

            sub12.set_ylabel('Power')
            sub12.plot(range(len(self.monthlyRemain['12'][:])), self.monthlyRemain['12'][:],color='green')  

        elif self.mode == 'load':
            ax1.plot(range(len(self.price)), self.price, label = "price")
            ax1.set_title('Jan')

            ax2.plot(range(len(self.price)), self.price, label = "price")
            ax2.set_title('Feb')

            ax3.plot(range(len(self.price)), self.price, label = "price")
            ax3.set_title('Mar')

            ax4.plot(range(len(self.price)), self.price, label = "price")
            ax4.set_title('Apr')

            ax5.plot(range(len(self.price)), self.price, label = "price")
            ax5.set_title('May')

            ax6.plot(range(len(self.price)), self.price, label = "price")
            ax6.set_title('Jun')

            ax7.plot(range(len(self.price)), self.price, label = "price")
            ax7.set_title('July')

            ax8.plot(range(len(self.price)), self.price, label = "price")
            ax8.set_title('Aug')

            ax9.plot(range(len(self.price)), self.price, label = "price")
            ax9.set_title('Sep')

            ax10.plot(range(len(self.price)), self.price, label = "price")
            ax10.set_title('Oct')

            ax11.plot(range(len(self.price)), self.price, label = "price")
            ax11.set_title('Nov')

            ax12.plot(range(len(self.price)), self.price, label = "price")
            ax12.set_title('Dec')

            sub1.set_ylabel('Power')
            sub1.bar(np.arange(95) ,self.monthlyRemain['1'][:] ,label = 'fixLoad',bottom = self.acConsume['1'][:]+self.wmConsume['1'][:] , color ='gray')  
            sub1.bar(np.arange(95) ,self.wmConsume['1'][:] ,label = 'WM',bottom = self.acConsume['1'][:], color ='orange')  
            sub1.bar(np.arange(95) ,self.acConsume['1'][:] ,label = 'AC', color ='green')  

            sub2.set_ylabel('Power')
            sub2.bar(np.arange(95) ,self.monthlyRemain['2'][:] ,label = 'fixLoad',bottom = self.acConsume['2'][:]+self.wmConsume['2'][:] , color ='gray')  
            sub2.bar(np.arange(95) ,self.wmConsume['2'][:] ,label = 'WM',bottom = self.acConsume['2'][:], color ='orange')  
            sub2.bar(np.arange(95) ,self.acConsume['2'][:] ,label = 'AC', color ='green')  

            sub3.set_ylabel('Power')
            sub3.bar(np.arange(95) ,self.monthlyRemain['3'][:] ,label = 'fixLoad',bottom = self.acConsume['3'][:]+self.wmConsume['3'][:] , color ='gray')  
            sub3.bar(np.arange(95) ,self.wmConsume['3'][:] ,label = 'WM',bottom = self.acConsume['3'][:], color ='orange')  
            sub3.bar(np.arange(95) ,self.acConsume['3'][:] ,label = 'AC', color ='green')  
            
            sub4.set_ylabel('Power')
            sub4.bar(np.arange(95) ,self.monthlyRemain['4'][:] ,label = 'fixLoad',bottom = self.acConsume['4'][:]+self.wmConsume['4'][:] , color ='gray')  
            sub4.bar(np.arange(95) ,self.wmConsume['4'][:] ,label = 'WM',bottom = self.acConsume['4'][:], color ='orange')  
            sub4.bar(np.arange(95) ,self.acConsume['4'][:] ,label = 'AC', color ='green')  
            
            sub5.set_ylabel('Power')
            sub5.bar(np.arange(95) ,self.monthlyRemain['5'][:] ,label = 'fixLoad',bottom = self.acConsume['5'][:]+self.wmConsume['5'][:] , color ='gray')  
            sub5.bar(np.arange(95) ,self.wmConsume['5'][:] ,label = 'WM',bottom = self.acConsume['5'][:], color ='orange')  
            sub5.bar(np.arange(95) ,self.acConsume['5'][:] ,label = 'AC', color ='green')  

            sub6.set_ylabel('Power')
            sub6.bar(np.arange(95) ,self.monthlyRemain['6'][:] ,label = 'fixLoad',bottom = self.acConsume['6'][:]+self.wmConsume['6'][:] , color ='gray')  
            sub6.bar(np.arange(95) ,self.wmConsume['6'][:] ,label = 'WM',bottom = self.acConsume['6'][:], color ='orange')  
            sub6.bar(np.arange(95) ,self.acConsume['6'][:] ,label = 'AC', color ='green')  
            

            
            sub7.set_ylabel('Power')
            sub7.bar(np.arange(95) ,self.monthlyRemain['7'][:] ,label = 'fixLoad',bottom = self.acConsume['7'][:]+self.wmConsume['7'][:] , color ='gray')  
            sub7.bar(np.arange(95) ,self.wmConsume['7'][:] ,label = 'WM',bottom = self.acConsume['7'][:], color ='orange')  
            sub7.bar(np.arange(95) ,self.acConsume['7'][:] ,label = 'AC', color ='green')  
            
            sub8.set_ylabel('Power')
            sub8.bar(np.arange(95) ,self.monthlyRemain['8'][:] ,label = 'fixLoad',bottom = self.acConsume['8'][:]+self.wmConsume['8'][:] , color ='gray')  
            sub8.bar(np.arange(95) ,self.wmConsume['8'][:] ,label = 'WM',bottom = self.acConsume['8'][:], color ='orange')  
            sub8.bar(np.arange(95) ,self.acConsume['8'][:] ,label = 'AC', color ='green')  

            sub9.set_ylabel('Power')
            sub9.bar(np.arange(95) ,self.monthlyRemain['9'][:] ,label = 'fixLoad',bottom = self.acConsume['9'][:]+self.wmConsume['9'][:] , color ='gray')  
            sub9.bar(np.arange(95) ,self.wmConsume['9'][:] ,label = 'WM',bottom = self.acConsume['9'][:], color ='orange')  
            sub9.bar(np.arange(95) ,self.acConsume['9'][:] ,label = 'AC', color ='green')  

            sub10.set_ylabel('Power')
            sub10.bar(np.arange(95) ,self.monthlyRemain['10'][:] ,label = 'fixLoad',bottom = self.acConsume['10'][:]+self.wmConsume['10'][:] , color ='gray')  
            sub10.bar(np.arange(95) ,self.wmConsume['10'][:] ,label = 'WM',bottom = self.acConsume['10'][:], color ='orange')  
            sub10.bar(np.arange(95) ,self.acConsume['10'][:] ,label = 'AC', color ='green')  

            sub11.set_ylabel('Power')
            sub11.bar(np.arange(95) ,self.monthlyRemain['11'][:] ,label = 'fixLoad',bottom = self.acConsume['11'][:]+self.wmConsume['11'][:] , color ='gray')  
            sub11.bar(np.arange(95) ,self.wmConsume['11'][:] ,label = 'WM',bottom = self.acConsume['11'][:], color ='orange')  
            sub11.bar(np.arange(95) ,self.acConsume['11'][:] ,label = 'AC', color ='green')  

            sub12.set_ylabel('Power')
            sub12.bar(np.arange(95) ,self.monthlyRemain['12'][:] ,label = 'fixLoad',bottom = self.acConsume['12'][:]+self.wmConsume['12'][:] , color ='gray')  
            sub12.bar(np.arange(95) ,self.wmConsume['12'][:] ,label = 'WM',bottom = self.acConsume['12'][:], color ='orange')  
            sub12.bar(np.arange(95) ,self.acConsume['12'][:] ,label = 'AC', color ='green')  

        fig.tight_layout()
        fig.savefig('pic/plot.png')


    def showNnConfig(self):
        network = self.agent.get_architecture()
        print(network)

    def __del__(self):
        # Close agent and environment
        self.agent.close()
        self.environment.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('please enter the mode: "soc" or "load"')
    test = Test(sys.argv[1])
    test.main()