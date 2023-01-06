from tensorforce import Agent,Environment
from lib.loads.interrupted import AC
from lib.loads.uninterrupted import WM
from lib.import_data.import_data import ImportData
from  yaml import load , SafeLoader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

class Test():
    def __init__(self,mode) :
        self.mode = mode
    def connetMysql(self):
        with open("yaml/mysqlData.yaml","r") as f:
            self.mysqlData = load(f,SafeLoader)
        self.host = self.mysqlData['host']
        self.user = self.mysqlData['user']
        self.passwd = self.mysqlData['passwd']
        self.db = self.mysqlData['db']
        self.info = ImportData(host= self.host ,user= self.user ,passwd= self.passwd ,db= self.db)
    def getGridPrice(self):
        self.allGridPrice = self.info.importGridPrice()
        self.summerGridPrice = self.allGridPrice['summer_price'].tolist()
        self.notSummerGridPrice = self.allGridPrice['not_summer_price'].tolist()
        self.testPrice = self.allGridPrice['test_price1'].tolist()
    def main(self):
    #run test result in different env
        if self.mode == 'soc':
            self.__testInSoc__()
        elif self.mode == 'intload':
            self.__testInInterruptibleLoad__()
        elif self.mode == 'unintload':
            self.__testUnInterruptibleLoad__()
        elif self.mode == 'HVAC':
            self.__testInHVAC__()

    #plot result
        self.__plotResult__()





    def __testInHVAC__(self):
        self.environment = Environment.create(environment='gym',level='Hems-v7')
        self.agent = Agent.load(directory = 'HVAC/saver_dir',environment=self.environment)
        load = []
        hvac = []
        pv = []
        indoorTemperature = []
        outdoorTemperature = []
        userSetTemperature = []
        totalReward = 0
        self.monthlyIndoorTemperature = pd.DataFrame()
        self.monthlyOutdoorTemperature = pd.DataFrame()
        self.monthlyRemain = pd.DataFrame()
        self.monthlyHVAC = pd.DataFrame()
        self.monthlyUserSetTemperature = pd.DataFrame()
        self.price = []
        for month in range(12):
            states = self.environment.reset()
            load.append(states[1])
            pv.append(states[2])
            indoorTemperature.append(states[4])
            outdoorTemperature.append(states[5])
            userSetTemperature.append(states[6])
            internals = self.agent.initial_internals()
            terminal = False
            while not terminal:
                actions, internals = self.agent.act(
                    states=states, internals=internals, independent=True, deterministic=True
                )
                states, terminal, reward = self.environment.execute(actions=actions)
                load.append(states[1])
                pv.append(states[2])
                if month == 11:
                    self.price.append(states[3])
                indoorTemperature.append(states[4])
                outdoorTemperature.append(states[5])
                userSetTemperature.append(states[6])
                hvac.append(actions[0])
                totalReward += reward

            hvac.append(0) # timestep = 96 , no action , but for plotting , all data should be size 96
            remain = [load[sampletime]-pv[sampletime] for sampletime in range(96)]
            #store testing result in each dictionary
            self.monthlyIndoorTemperature.insert(month,column=str(month+1),value=indoorTemperature)
            self.monthlyOutdoorTemperature.insert(month,column=str(month+1),value=outdoorTemperature)
            self.monthlyRemain.insert(month,column=str(month+1),value=remain)
            self.monthlyHVAC.insert(month,column=str(month+1),value=hvac)
            self.monthlyUserSetTemperature.insert(month,column=str(month+1),value=userSetTemperature)
            load.clear()
            pv.clear()
            indoorTemperature.clear()
            outdoorTemperature.clear()
            userSetTemperature.clear()
            hvac.clear()
        print('Agent average episode reward: ', totalReward/12 )    


    def __testInSoc__(self):
        self.environment = Environment.create(environment='gym',level='Hems-v1')
        self.agent = Agent.load(directory = 'Soc/saver_dir',environment=self.environment)
        soc = []
        load = []
        socPower = []
        pv = []
        degradation = []
        totalReward = 0
        self.monthlySoc = pd.DataFrame()
        self.monthlySocPower = pd.DataFrame()
        self.monthlyRemain = pd.DataFrame()
        self.monthlyDegradation = pd.DataFrame()
        self.price = []
        for month in range(12):
            states = self.environment.reset()
            load.append(states[1])
            pv.append(states[2])
            soc.append(states[3])
            degradation.append(states[5])
            internals = self.agent.initial_internals()
            terminal = False
            while not terminal:
                print(states)
                actions, internals = self.agent.act(
                    states=states, internals=internals, independent=True, deterministic=True
                )
                states, terminal, reward = self.environment.execute(actions=actions)
                socPower.append(actions[0]*10)
                load.append(states[1])
                pv.append(states[2])
                soc.append(states[3])
                degradation.append(states[5])
                totalReward += reward
                if month == 11:
                    self.price.append(states[4])

            socPower.append(0)
            remain = [load[sampletime]-pv[sampletime] for sampletime in range(96)]
            self.monthlySoc.insert(month,column=str(month+1),value=soc)
            self.monthlySocPower.insert(month,column=str(month+1),value=socPower)
            self.monthlyRemain.insert(month,column=str(month+1),value=remain)
            self.monthlyDegradation.insert(month,column=str(month+1),value=degradation)

            load.clear()
            pv.clear()
            soc.clear()
            socPower.clear()
            degradation.clear()
        print('Agent average episode reward: ', totalReward/12 )

    def __testInInterruptibleLoad__(self):
        self.environment = Environment.create(environment='gym',level='Hems-v5')
        self.agent = Agent.load(directory = 'Load/Interruptible/saver_dir',environment=self.environment)
        ac_object = AC(demand=40,AvgPowerConsume=0.3)
        load = []
        pv = []
        ac = []
        totalReward = 0
        self.monthlyRemain = pd.DataFrame()
        self.acConsume = pd.DataFrame()
        self.price = []
        for month in range(12):
            states = self.environment.reset()
            load.append(states[1])
            pv.append(states[2])
            internals = self.agent.initial_internals()
            terminal = False
            while not terminal:
                actions, internals = self.agent.act(
                    states=states, internals=internals, independent=True, deterministic=True
                )
                states, terminal, reward = self.environment.execute(actions=actions)
                #1. AC on 
                if actions == 0 and ac_object.getRemainDemand()>0:
                    ac_object.turn_on()
                    ac.append(ac_object.AvgPowerConsume)#power
                #2. AC off 
                else :
                    ac_object.turn_off()
                    ac.append(0)

                load.append(states[1])
                pv.append(states[2])
                totalReward += reward
                if month == 11:
                    self.price.append(states[3])

            ac.append(0)
            remain = [load[sampletime]-pv[sampletime] for sampletime in range(96)]
            #normalize price to [0,1]
            self.monthlyRemain.insert(month,column=str(month+1),value=remain)
            self.acConsume.insert(month,column=str(month+1),value=ac)
            ac_object.reset()
            ac.clear()
            load.clear()
            pv.clear()
        print('Agent average episode reward: ', totalReward/12 )

    def __testUnInterruptibleLoad__(self,mode='normal'):
        self.connetMysql()
        self.getGridPrice()
        summer = [5,6,7,8]
        self.environment = Environment.create(environment='gym',level='Hems-v9')
        self.agent = Agent.load(directory = 'Load/UnInterruptible/saver_dir',environment=self.environment)
        wmObject = WM(demand=6,executePeriod=8,AvgPowerConsume=0.3)
        load = []
        pv = []
        wm = []
        totalReward = 0
        self.monthlyRemain = pd.DataFrame()
        self.wmConsume = pd.DataFrame()
        self.price = pd.DataFrame()
        for month in range(12):
            states = self.environment.reset()
            load.append(states[1])
            pv.append(states[2])
            internals = self.agent.initial_internals()
            terminal = False
            while not terminal:
                actions, internals = self.agent.act(
                    states=states, internals=internals, independent=True, deterministic=True
                )
                states, terminal, reward = self.environment.execute(actions=actions)
                #print(states)
                #1. WM on 
                if states[5] == 1: # washing machine's switch
                    wm.append(wmObject.AvgPowerConsume)#power
                #2. do nothing 
                else :
                    wm.append(0)

                load.append(states[1])
                pv.append(states[2])
                totalReward += reward
            if mode == 'normal':
                if month not in summer :
                    self.price.insert(month,column = str(month+1),value=self.notSummerGridPrice)
                else :
                    self.price.insert(month,column = str(month+1),value=self.summerGridPrice)
            else:
                self.price.insert(month,column = str(month+1),value=self.testPrice)
                


            wm.append(0)
            remain = [load[sampletime]-pv[sampletime] for sampletime in range(96)]
            #normalize price to [0,1]
            self.monthlyRemain.insert(month,column=str(month+1),value=remain)
            self.wmConsume.insert(month,column=str(month+1),value=wm)
            wmObject.reset()
            wm.clear()
            load.clear()
            pv.clear()
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

        if self.mode == 'soc':
            ax1.set_ylabel('SOC')
            ax1.set_ylim(0,1)
            ax1.plot(range(len(self.monthlySoc['1'][:])), self.monthlySoc['1'][:], label = "Jan",color='red')    
            ax1.set_title('Jan')

            ax2.set_ylabel('SOC')
            ax2.set_ylim(0,1)
            ax2.plot(range(len(self.monthlySoc['2'][:])), self.monthlySoc['2'][:], label = "Feb",color='red')
            ax2.set_title('Feb')

            ax3.set_ylabel('SOC')
            ax3.set_ylim(0,1)
            ax3.plot(range(len(self.monthlySoc['3'][:])), self.monthlySoc['3'][:], label = "Mar",color='red')
            ax3.set_title('Mar')

            ax4.set_ylabel('SOC')
            ax4.set_ylim(0,1)
            ax4.plot(range(len(self.monthlySoc['4'][:])), self.monthlySoc['4'][:], label = "Apr",color='red')
            ax4.set_title('Apr')

            ax5.set_ylabel('SOC')
            ax5.set_ylim(0,1)
            ax5.plot(range(len(self.monthlySoc['5'][:])), self.monthlySoc['5'][:], label = "May",color='red')
            ax5.set_title('May')

            ax6.set_ylabel('SOC')
            ax6.set_ylim(0,1)
            ax6.plot(range(len(self.monthlySoc['6'][:])), self.monthlySoc['6'][:], label = "Jun",color='red')
            ax6.set_title('Jun')

            ax7.set_ylabel('SOC')
            ax7.set_ylim(0,1)
            ax7.plot(range(len(self.monthlySoc['7'][:])), self.monthlySoc['7'][:], label = "July",color='red')
            ax7.set_title('July')

            ax8.set_ylabel('SOC')
            ax8.set_ylim(0,1)
            ax8.plot(range(len(self.monthlySoc['8'][:])), self.monthlySoc['8'][:], label = "Aug",color='red')
            ax8.set_title('Aug')

            ax9.set_ylabel('SOC')
            ax9.set_ylim(0,1)
            ax9.plot(range(len(self.monthlySoc['9'][:])), self.monthlySoc['9'][:], label = "Sep",color='red')
            ax9.set_title('Sep')

            ax10.set_ylabel('SOC')
            ax10.set_ylim(0,1)
            ax10.plot(range(len(self.monthlySoc['10'][:])), self.monthlySoc['10'][:], label = "Oct",color='red')
            ax10.set_title('Oct')

            ax11.set_ylabel('SOC')
            ax11.set_ylim(0,1)
            ax11.plot(range(len(self.monthlySoc['11'][:])), self.monthlySoc['11'][:], label = "Nov",color='red')
            ax11.set_title('Nov')

            ax12.set_ylabel('SOC')
            ax12.set_ylim(0,1)
            ax12.plot(range(len(self.monthlySoc['12'][:])), self.monthlySoc['12'][:], label = "Dec",color='red')
            ax12.set_title('Dec')

            sub1.set_ylabel('Power')
            sub1.bar(np.arange(96) ,self.monthlyRemain['1'][:] ,label = 'fixLoad',bottom = self.monthlySocPower['1'][:] , color ='gray')  
            sub1.bar(np.arange(96) ,self.monthlySocPower['1'][:] ,label = 'socPower',color ='red')  

            sub2.set_ylabel('Power')
            sub2.bar(np.arange(96) ,self.monthlyRemain['2'][:] ,label = 'fixLoad',bottom = self.monthlySocPower['2'][:] , color ='gray')  
            sub2.bar(np.arange(96) ,self.monthlySocPower['2'][:] ,label = 'socPower',color ='red')  

            sub3.set_ylabel('Power')
            sub3.bar(np.arange(96) ,self.monthlyRemain['3'][:] ,label = 'fixLoad',bottom = self.monthlySocPower['3'][:] , color ='gray')  
            sub3.bar(np.arange(96) ,self.monthlySocPower['3'][:] ,label = 'socPower',color ='red')  

            sub4.set_ylabel('Power')
            sub4.bar(np.arange(96) ,self.monthlyRemain['4'][:] ,label = 'fixLoad',bottom = self.monthlySocPower['4'][:] , color ='gray')  
            sub4.bar(np.arange(96) ,self.monthlySocPower['4'][:] ,label = 'socPower',color ='red')  

            sub5.set_ylabel('Power')
            sub5.bar(np.arange(96) ,self.monthlyRemain['5'][:] ,label = 'fixLoad',bottom = self.monthlySocPower['5'][:] , color ='gray')  
            sub5.bar(np.arange(96) ,self.monthlySocPower['5'][:] ,label = 'socPower',color ='red')  

            sub6.set_ylabel('Power')
            sub6.bar(np.arange(96) ,self.monthlyRemain['6'][:] ,label = 'fixLoad',bottom = self.monthlySocPower['6'][:] , color ='gray')  
            sub6.bar(np.arange(96) ,self.monthlySocPower['6'][:] ,label = 'socPower',color ='red')  

            sub7.set_ylabel('Power')
            sub7.bar(np.arange(96) ,self.monthlyRemain['7'][:] ,label = 'fixLoad',bottom = self.monthlySocPower['7'][:] , color ='gray')  
            sub7.bar(np.arange(96) ,self.monthlySocPower['7'][:] ,label = 'socPower',color ='red')  

            sub8.set_ylabel('Power')
            sub8.bar(np.arange(96) ,self.monthlyRemain['8'][:] ,label = 'fixLoad',bottom = self.monthlySocPower['8'][:] , color ='gray')  
            sub8.bar(np.arange(96) ,self.monthlySocPower['8'][:] ,label = 'socPower',color ='red')  

            sub9.set_ylabel('Power')
            sub9.bar(np.arange(96) ,self.monthlyRemain['9'][:] ,label = 'fixLoad',bottom = self.monthlySocPower['9'][:] , color ='gray')  
            sub9.bar(np.arange(96) ,self.monthlySocPower['9'][:] ,label = 'socPower',color ='red')  

            sub10.set_ylabel('Power')
            sub10.bar(np.arange(96) ,self.monthlyRemain['10'][:] ,label = 'fixLoad',bottom = self.monthlySocPower['10'][:] , color ='gray')  
            sub10.bar(np.arange(96) ,self.monthlySocPower['10'][:] ,label = 'socPower', color ='red')  

            sub11.set_ylabel('Power')
            sub11.bar(np.arange(96) ,self.monthlyRemain['11'][:] ,label = 'fixLoad',bottom = self.monthlySocPower['11'][:] , color ='gray')  
            sub11.bar(np.arange(96) ,self.monthlySocPower['11'][:] ,label = 'socPower', color ='red')  

            sub12.set_ylabel('Power')
            sub12.bar(np.arange(96) ,self.monthlyRemain['12'][:] ,label = 'fixLoad',bottom = self.monthlySocPower['12'][:] , color ='gray')  
            sub12.bar(np.arange(96) ,self.monthlySocPower['12'][:] ,label = 'socPower', color ='red') 

            sub1a = ax1.twinx()
            sub1a.spines['right'].set_position(("axes",1.1))
            sub1a.tick_params(axis='y',colors = "blue")
            sub1a.set_ylabel('price',color='blue')
            sub1a.plot(range(len(self.price)), self.price, label = "price")
            sub1a.plot(range(len(self.monthlyDegradation['1'][:])), self.monthlyDegradation['1'][:], label = "degradation")

            sub2a = ax2.twinx()
            sub2a.spines['right'].set_position(("axes",1.1))
            sub2a.tick_params(axis='y',colors = "blue")
            sub2a.set_ylabel('price',color='blue')
            sub2a.plot(range(len(self.price)), self.price, label = "price")
            sub2a.plot(range(len(self.monthlyDegradation['2'][:])), self.monthlyDegradation['2'][:], label = "degradation")

            sub3a = ax3.twinx()
            sub3a.spines['right'].set_position(("axes",1.1))
            sub3a.tick_params(axis='y',colors = "blue")
            sub3a.set_ylabel('price',color='blue')
            sub3a.plot(range(len(self.price)), self.price, label = "price")
            sub3a.plot(range(len(self.monthlyDegradation['3'][:])), self.monthlyDegradation['3'][:], label = "degradation")

            sub4a = ax4.twinx()
            sub4a.spines['right'].set_position(("axes",1.1))
            sub4a.tick_params(axis='y',colors = "blue")
            sub4a.set_ylabel('price',color='blue')
            sub4a.plot(range(len(self.price)), self.price, label = "price")
            sub4a.plot(range(len(self.monthlyDegradation['4'][:])), self.monthlyDegradation['4'][:], label = "degradation")

            sub5a = ax5.twinx()
            sub5a.spines['right'].set_position(("axes",1.1))
            sub5a.tick_params(axis='y',colors = "blue")
            sub5a.set_ylabel('price',color='blue')
            sub5a.plot(range(len(self.price)), self.price, label = "price")
            sub5a.plot(range(len(self.monthlyDegradation['5'][:])), self.monthlyDegradation['5'][:], label = "degradation")

            sub6a = ax6.twinx()
            sub6a.spines['right'].set_position(("axes",1.1))
            sub6a.tick_params(axis='y',colors = "blue")
            sub6a.set_ylabel('price',color='blue')
            sub6a.plot(range(len(self.price)), self.price, label = "price")
            sub6a.plot(range(len(self.monthlyDegradation['6'][:])), self.monthlyDegradation['6'][:], label = "degradation")

            sub7a = ax7.twinx()
            sub7a.spines['right'].set_position(("axes",1.1))
            sub7a.tick_params(axis='y',colors = "blue")
            sub7a.set_ylabel('price',color='blue')
            sub7a.plot(range(len(self.price)), self.price, label = "price")
            sub7a.plot(range(len(self.monthlyDegradation['7'][:])), self.monthlyDegradation['7'][:], label = "degradation")

            sub8a = ax8.twinx()
            sub8a.spines['right'].set_position(("axes",1.1))
            sub8a.tick_params(axis='y',colors = "blue")
            sub8a.set_ylabel('price',color='blue')
            sub8a.plot(range(len(self.price)), self.price, label = "price")
            sub8a.plot(range(len(self.monthlyDegradation['8'][:])), self.monthlyDegradation['8'][:], label = "degradation")

            sub9a = ax9.twinx()
            sub9a.spines['right'].set_position(("axes",1.1))
            sub9a.tick_params(axis='y',colors = "blue")
            sub9a.set_ylabel('price',color='blue')
            sub9a.plot(range(len(self.price)), self.price, label = "price")
            sub9a.plot(range(len(self.monthlyDegradation['9'][:])), self.monthlyDegradation['9'][:], label = "degradation")

            sub10a = ax10.twinx()
            sub10a.spines['right'].set_position(("axes",1.1))
            sub10a.tick_params(axis='y',colors = "blue")
            sub10a.set_ylabel('price',color='blue')
            sub10a.plot(range(len(self.price)), self.price, label = "price")
            sub10a.plot(range(len(self.monthlyDegradation['10'][:])), self.monthlyDegradation['10'][:], label = "degradation")

            sub11a = ax11.twinx()
            sub11a.spines['right'].set_position(("axes",1.1))
            sub11a.tick_params(axis='y',colors = "blue")
            sub11a.set_ylabel('price',color='blue')
            sub11a.plot(range(len(self.price)), self.price, label = "price")
            sub11a.plot(range(len(self.monthlyDegradation['11'][:])), self.monthlyDegradation['11'][:], label = "degradation")

            sub12a = ax12.twinx()
            sub12a.spines['right'].set_position(("axes",1.1))
            sub12a.tick_params(axis='y',colors = "blue")
            sub12a.set_ylabel('price',color='blue')
            sub12a.plot(range(len(self.price)), self.price, label = "price")
            sub12a.plot(range(len(self.monthlyDegradation['12'][:])), self.monthlyDegradation['12'][:], label = "degradation")

            fig.tight_layout()
            fig.savefig('pic/SOC/newestSocResult.png') 

        elif self.mode == 'intload':
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
            sub1.bar(np.arange(96) ,self.monthlyRemain['1'][:] ,label = 'fixLoad',bottom = self.acConsume['1'][:] , color ='gray')  
            sub1.bar(np.arange(96) ,self.acConsume['1'][:] ,label = 'AC', color ='green')  

            sub2.set_ylabel('Power')
            sub2.bar(np.arange(96) ,self.monthlyRemain['2'][:] ,label = 'fixLoad',bottom = self.acConsume['2'][:] , color ='gray')  
            sub2.bar(np.arange(96) ,self.acConsume['2'][:] ,label = 'AC', color ='green')  

            sub3.set_ylabel('Power')
            sub3.bar(np.arange(96) ,self.monthlyRemain['3'][:] ,label = 'fixLoad',bottom = self.acConsume['3'][:] , color ='gray')  
            sub3.bar(np.arange(96) ,self.acConsume['3'][:] ,label = 'AC', color ='green')  
            
            sub4.set_ylabel('Power')
            sub4.bar(np.arange(96) ,self.monthlyRemain['4'][:] ,label = 'fixLoad',bottom = self.acConsume['4'][:] , color ='gray')  
            sub4.bar(np.arange(96) ,self.acConsume['4'][:] ,label = 'AC', color ='green')  
            
            sub5.set_ylabel('Power')
            sub5.bar(np.arange(96) ,self.monthlyRemain['5'][:] ,label = 'fixLoad',bottom = self.acConsume['5'][:] , color ='gray')  
            sub5.bar(np.arange(96) ,self.acConsume['5'][:] ,label = 'AC', color ='green')  

            sub6.set_ylabel('Power')
            sub6.bar(np.arange(96) ,self.monthlyRemain['6'][:] ,label = 'fixLoad',bottom = self.acConsume['6'][:] , color ='gray')  
            sub6.bar(np.arange(96) ,self.acConsume['6'][:] ,label = 'AC', color ='green')  

            sub7.set_ylabel('Power')
            sub7.bar(np.arange(96) ,self.monthlyRemain['7'][:] ,label = 'fixLoad',bottom = self.acConsume['7'][:] , color ='gray')  
            sub7.bar(np.arange(96) ,self.acConsume['7'][:] ,label = 'AC', color ='green')  
            
            sub8.set_ylabel('Power')
            sub8.bar(np.arange(96) ,self.monthlyRemain['8'][:] ,label = 'fixLoad',bottom = self.acConsume['8'][:] , color ='gray')  
            sub8.bar(np.arange(96) ,self.acConsume['8'][:] ,label = 'AC', color ='green')  

            sub9.set_ylabel('Power')
            sub9.bar(np.arange(96) ,self.monthlyRemain['9'][:] ,label = 'fixLoad',bottom = self.acConsume['9'][:], color ='gray')  
            sub9.bar(np.arange(96) ,self.acConsume['9'][:] ,label = 'AC', color ='green')  

            sub10.set_ylabel('Power')
            sub10.bar(np.arange(96) ,self.monthlyRemain['10'][:] ,label = 'fixLoad',bottom = self.acConsume['10'][:] , color ='gray')  
            sub10.bar(np.arange(96) ,self.acConsume['10'][:] ,label = 'AC', color ='green')  

            sub11.set_ylabel('Power')
            sub11.bar(np.arange(96) ,self.monthlyRemain['11'][:] ,label = 'fixLoad',bottom = self.acConsume['11'][:], color ='gray')  
            sub11.bar(np.arange(96) ,self.acConsume['11'][:] ,label = 'AC', color ='green')  

            sub12.set_ylabel('Power')
            sub12.bar(np.arange(96) ,self.monthlyRemain['12'][:] ,label = 'fixLoad',bottom = self.acConsume['12'][:], color ='gray')  
            sub12.bar(np.arange(96) ,self.acConsume['12'][:] ,label = 'AC', color ='green') 

            fig.tight_layout()
            fig.savefig('pic/Loads/newestIntLoadsResult.png') 

        elif self.mode == 'unintload':
            ax1.plot(range(len(self.price)), self.price['1'][:], label = "price")
            ax1.set_title('Jan')
            ax1.set_ylim(0,6.5)


            ax2.plot(range(len(self.price)), self.price['2'][:], label = "price")
            ax2.set_title('Feb')
            ax2.set_ylim(0,6.5)

            ax3.plot(range(len(self.price)), self.price['3'][:], label = "price")
            ax3.set_title('Mar')
            ax3.set_ylim(0,6.5)

            ax4.plot(range(len(self.price)), self.price['4'][:], label = "price")
            ax4.set_title('Apr')
            ax4.set_ylim(0,6.5)

            ax5.plot(range(len(self.price)), self.price['5'][:], label = "price")
            ax5.set_title('May')
            ax5.set_ylim(0,6.5)

            ax6.plot(range(len(self.price)), self.price['6'][:], label = "price")
            ax6.set_title('Jun')
            ax6.set_ylim(0,6.5)

            ax7.plot(range(len(self.price)), self.price['7'][:], label = "price")
            ax7.set_title('July')
            ax7.set_ylim(0,6.5)

            ax8.plot(range(len(self.price)), self.price['8'][:], label = "price")
            ax8.set_title('Aug')
            ax8.set_ylim(0,6.5)

            ax9.plot(range(len(self.price)), self.price['9'][:], label = "price")
            ax9.set_title('Sep')
            ax9.set_ylim(0,6.5)

            ax10.plot(range(len(self.price)), self.price['10'][:], label = "price")
            ax10.set_title('Oct')
            ax10.set_ylim(0,6.5)

            ax11.plot(range(len(self.price)), self.price['11'][:], label = "price")
            ax11.set_title('Nov')
            ax11.set_ylim(0,6.5)

            ax12.plot(range(len(self.price)), self.price['12'][:], label = "price")
            ax12.set_title('Dec')
            ax12.set_ylim(0,6.5)

            sub1.set_ylabel('Power')
            sub1.bar(np.arange(96) ,self.monthlyRemain['1'][:] ,label = 'fixLoad',bottom = self.wmConsume['1'][:] , color ='gray')  
            sub1.bar(np.arange(96) ,self.wmConsume['1'][:] ,label = 'WM', color ='green')  
            #sub1.bar(np.arange(96) ,self.acConsume['1'][:] ,label = 'AC', color ='gray')  

            sub2.set_ylabel('Power')
            sub2.bar(np.arange(96) ,self.monthlyRemain['2'][:] ,label = 'fixLoad',bottom = self.wmConsume['2'][:] , color ='gray')  
            sub2.bar(np.arange(96) ,self.wmConsume['2'][:] ,label = 'WM', color ='green')  
            #sub2.bar(np.arange(96) ,self.acConsume['2'][:] ,label = 'AC', color ='gray')  

            sub3.set_ylabel('Power')
            sub3.bar(np.arange(96) ,self.monthlyRemain['3'][:] ,label = 'fixLoad',bottom = self.wmConsume['3'][:] , color ='gray')  
            sub3.bar(np.arange(96) ,self.wmConsume['3'][:] ,label = 'WM', color ='green')  
            #sub3.bar(np.arange(96) ,self.acConsume['3'][:] ,label = 'AC', color ='gray')  
            
            sub4.set_ylabel('Power')
            sub4.bar(np.arange(96) ,self.monthlyRemain['4'][:] ,label = 'fixLoad',bottom = self.wmConsume['4'][:] , color ='gray')  
            sub4.bar(np.arange(96) ,self.wmConsume['4'][:] ,label = 'WM', color ='green')  
            #sub4.bar(np.arange(96) ,self.acConsume['4'][:] ,label = 'AC', color ='gray')  
            
            sub5.set_ylabel('Power')
            sub5.bar(np.arange(96) ,self.monthlyRemain['5'][:] ,label = 'fixLoad',bottom = self.wmConsume['5'][:] , color ='gray')  
            sub5.bar(np.arange(96) ,self.wmConsume['5'][:] ,label = 'WM', color ='green')  
            #sub5.bar(np.arange(96) ,self.acConsume['5'][:] ,label = 'AC', color ='gray')  

            sub6.set_ylabel('Power')
            sub6.bar(np.arange(96) ,self.monthlyRemain['6'][:] ,label = 'fixLoad',bottom = self.wmConsume['6'][:] , color ='gray')  
            sub6.bar(np.arange(96) ,self.wmConsume['6'][:] ,label = 'WM', color ='green')  
            #sub6.bar(np.arange(96) ,self.acConsume['6'][:] ,label = 'AC', color ='gray')  
    
            
            sub7.set_ylabel('Power')
            sub7.bar(np.arange(96) ,self.monthlyRemain['7'][:] ,label = 'fixLoad',bottom = self.wmConsume['7'][:] , color ='gray')  
            sub7.bar(np.arange(96) ,self.wmConsume['7'][:] ,label = 'WM', color ='green')  
            #sub7.bar(np.arange(96) ,self.acConsume['7'][:] ,label = 'AC', color ='gray')  
            
            sub8.set_ylabel('Power')
            sub8.bar(np.arange(96) ,self.monthlyRemain['8'][:] ,label = 'fixLoad',bottom = self.wmConsume['8'][:] , color ='gray')  
            sub8.bar(np.arange(96) ,self.wmConsume['8'][:] ,label = 'WM', color ='green')  
            #sub8.bar(np.arange(96) ,self.acConsume['8'][:] ,label = 'AC', color ='gray')  

            sub9.set_ylabel('Power')
            sub9.bar(np.arange(96) ,self.monthlyRemain['9'][:] ,label = 'fixLoad',bottom = self.wmConsume['9'][:], color ='gray')  
            sub9.bar(np.arange(96) ,self.wmConsume['9'][:] ,label = 'WM', color ='green')  
            #sub9.bar(np.arange(96) ,self.acConsume['9'][:] ,label = 'AC', color ='gray')  

            sub10.set_ylabel('Power')
            sub10.bar(np.arange(96) ,self.monthlyRemain['10'][:] ,label = 'fixLoad',bottom = self.wmConsume['10'][:] , color ='gray')  
            sub10.bar(np.arange(96) ,self.wmConsume['10'][:] ,label = 'WM', color ='green')  
            #sub10.bar(np.arange(96) ,self.acConsume['10'][:] ,label = 'AC', color ='gray')  

            sub11.set_ylabel('Power')
            sub11.bar(np.arange(96) ,self.monthlyRemain['11'][:] ,label = 'fixLoad',bottom = self.wmConsume['11'][:], color ='gray')  
            sub11.bar(np.arange(96) ,self.wmConsume['11'][:] ,label = 'WM', color ='green')  
            #sub11.bar(np.arange(96) ,self.acConsume['11'][:] ,label = 'AC', color ='gray')  

            sub12.set_ylabel('Power')
            sub12.bar(np.arange(96) ,self.monthlyRemain['12'][:] ,label = 'fixLoad',bottom = self.wmConsume['12'][:], color ='gray')  
            sub12.bar(np.arange(96) ,self.wmConsume['12'][:] ,label = 'WM', color ='green')  
            #sub12.bar(np.arange(96) ,self.acConsume['12'][:] ,label = 'AC', color ='gray') 

            fig.tight_layout()
            fig.savefig('pic/Loads/uninterruptible/newestUnIntLoadsResult.png') 

        elif self.mode == 'HVAC':
            
            #transfer indoorTemperature unit
            #transfer outdoorTemperature unit
            self.monthlyIndoorTemperature = (self.monthlyIndoorTemperature-32) *5/9
            self.monthlyOutdoorTemperature = (self.monthlyOutdoorTemperature-32) *5/9
            self.monthlyUserSetTemperature = (self.monthlyUserSetTemperature-32) *5/9
            

            #more twinx

            
            ax1.set_ylabel('Temperature')
            ax1.set_ylim(10,45)
            ax1.plot(range(len(self.monthlyIndoorTemperature['1'][:])), self.monthlyIndoorTemperature['1'][:], label = "Jan",color='red')    
            ax1.plot(range(len(self.monthlyOutdoorTemperature['1'][:])), self.monthlyOutdoorTemperature['1'][:], label = "Jan",color='orange')    
            ax1.plot(range(len(self.monthlyUserSetTemperature['1'][:])), self.monthlyUserSetTemperature['1'][:], label = "Jan",color='black')    
            ax1.set_title('Jan')

            ax2.set_ylabel('Temperature')
            ax2.set_ylim(10,45)
            ax2.plot(range(len(self.monthlyIndoorTemperature['2'][:])), self.monthlyIndoorTemperature['2'][:], label = "Feb",color='red')
            ax2.plot(range(len(self.monthlyOutdoorTemperature['2'][:])), self.monthlyOutdoorTemperature['2'][:], label = "Feb",color='orange')
            ax2.plot(range(len(self.monthlyUserSetTemperature['2'][:])), self.monthlyUserSetTemperature['2'][:], label = "Feb",color='black')
            ax2.set_title('Feb')

            ax3.set_ylabel('Temperature')
            ax3.set_ylim(10,45)
            ax3.plot(range(len(self.monthlyIndoorTemperature['3'][:])), self.monthlyIndoorTemperature['3'][:], label = "Mar",color='red')
            ax3.plot(range(len(self.monthlyOutdoorTemperature['3'][:])), self.monthlyOutdoorTemperature['3'][:], label = "Mar",color='orange')
            ax3.plot(range(len(self.monthlyUserSetTemperature['3'][:])), self.monthlyUserSetTemperature['3'][:], label = "Mar",color='black')
            ax3.set_title('Mar')

            ax4.set_ylabel('Temperature')
            ax4.set_ylim(10,45)
            ax4.plot(range(len(self.monthlyIndoorTemperature['4'][:])), self.monthlyIndoorTemperature['4'][:], label = "Apr",color='red')
            ax4.plot(range(len(self.monthlyOutdoorTemperature['4'][:])), self.monthlyOutdoorTemperature['4'][:], label = "Apr",color='orange')
            ax4.plot(range(len(self.monthlyUserSetTemperature['4'][:])), self.monthlyUserSetTemperature['4'][:], label = "Apr",color='black')
            ax4.set_title('Apr')

            ax5.set_ylabel('Temperature')
            ax5.set_ylim(10,45)
            ax5.plot(range(len(self.monthlyIndoorTemperature['5'][:])), self.monthlyIndoorTemperature['5'][:], label = "May",color='red')
            ax5.plot(range(len(self.monthlyOutdoorTemperature['5'][:])), self.monthlyOutdoorTemperature['5'][:], label = "May",color='orange')
            ax5.plot(range(len(self.monthlyUserSetTemperature['5'][:])), self.monthlyUserSetTemperature['5'][:], label = "May",color='black')
            ax5.set_title('May')

            ax6.set_ylabel('Temperature')
            ax6.set_ylim(10,45)
            ax6.plot(range(len(self.monthlyIndoorTemperature['6'][:])), self.monthlyIndoorTemperature['6'][:], label = "Jun",color='red')
            ax6.plot(range(len(self.monthlyOutdoorTemperature['6'][:])), self.monthlyOutdoorTemperature['6'][:], label = "Jun",color='orange')
            ax6.plot(range(len(self.monthlyUserSetTemperature['6'][:])), self.monthlyUserSetTemperature['6'][:], label = "Jun",color='black')
            ax6.set_title('Jun')

            ax7.set_ylabel('Temperature')
            ax7.set_ylim(10,45)
            ax7.plot(range(len(self.monthlyIndoorTemperature['7'][:])), self.monthlyIndoorTemperature['7'][:], label = "July",color='red')
            ax7.plot(range(len(self.monthlyOutdoorTemperature['7'][:])), self.monthlyOutdoorTemperature['7'][:], label = "July",color='orange')
            ax7.plot(range(len(self.monthlyUserSetTemperature['7'][:])), self.monthlyUserSetTemperature['7'][:], label = "July",color='black')
            ax7.set_title('July')

            ax8.set_ylabel('Temperature')
            ax8.set_ylim(10,45)
            ax8.plot(range(len(self.monthlyIndoorTemperature['8'][:])), self.monthlyIndoorTemperature['8'][:], label = "Aug",color='red')
            ax8.plot(range(len(self.monthlyOutdoorTemperature['8'][:])), self.monthlyOutdoorTemperature['8'][:], label = "Aug",color='orange')
            ax8.plot(range(len(self.monthlyUserSetTemperature['8'][:])), self.monthlyUserSetTemperature['8'][:], label = "Aug",color='black')
            ax8.set_title('Aug')

            ax9.set_ylabel('Temperature')
            ax9.set_ylim(10,45)
            ax9.plot(range(len(self.monthlyIndoorTemperature['9'][:])), self.monthlyIndoorTemperature['9'][:], label = "Sep",color='red')
            ax9.plot(range(len(self.monthlyOutdoorTemperature['9'][:])), self.monthlyOutdoorTemperature['9'][:], label = "Sep",color='orange')
            ax9.plot(range(len(self.monthlyUserSetTemperature['9'][:])), self.monthlyUserSetTemperature['9'][:], label = "Sep",color='black')
            ax9.set_title('Sep')

            ax10.set_ylabel('Temperature')
            ax10.set_ylim(10,45)
            ax10.plot(range(len(self.monthlyIndoorTemperature['10'][:])), self.monthlyIndoorTemperature['10'][:], label = "Oct",color='red')
            ax10.plot(range(len(self.monthlyOutdoorTemperature['10'][:])), self.monthlyOutdoorTemperature['10'][:], label = "Oct",color='orange')
            ax10.plot(range(len(self.monthlyUserSetTemperature['10'][:])), self.monthlyUserSetTemperature['10'][:], label = "Oct",color='black')
            ax10.set_title('Oct')

            ax11.set_ylabel('Temperature')
            ax11.set_ylim(10,45)
            ax11.plot(range(len(self.monthlyIndoorTemperature['11'][:])), self.monthlyIndoorTemperature['11'][:], label = "Nov",color='red')
            ax11.plot(range(len(self.monthlyOutdoorTemperature['11'][:])), self.monthlyOutdoorTemperature['11'][:], label = "Nov",color='orange')
            ax11.plot(range(len(self.monthlyUserSetTemperature['11'][:])), self.monthlyUserSetTemperature['11'][:], label = "Nov",color='black')
            ax11.set_title('Nov')

            ax12.set_ylabel('Temperature')
            ax12.set_ylim(10,45)
            ax12.plot(range(len(self.monthlyIndoorTemperature['12'][:])), self.monthlyIndoorTemperature['12'][:], label = "Dec",color='red')
            ax12.plot(range(len(self.monthlyOutdoorTemperature['12'][:])), self.monthlyOutdoorTemperature['12'][:], label = "Dec",color='orange')
            ax12.plot(range(len(self.monthlyUserSetTemperature['12'][:])), self.monthlyUserSetTemperature['12'][:], label = "Dec",color='black')
            ax12.set_title('Dec')

#-----------------------------------------------------------------------------------------------#

            sub1.set_ylabel('Power')
            sub1.bar(np.arange(96) ,self.monthlyRemain['1'][:] ,label = 'fixLoad',bottom = self.monthlyHVAC['1'][:] , color ='gray')  
            sub1.bar(np.arange(96) ,self.monthlyHVAC['1'][:] ,label = 'HVAC',color ='red')  

            sub2.set_ylabel('Power')
            sub2.bar(np.arange(96) ,self.monthlyRemain['2'][:] ,label = 'fixLoad',bottom = self.monthlyHVAC['2'][:] , color ='gray')  
            sub2.bar(np.arange(96) ,self.monthlyHVAC['2'][:] ,label = 'HVAC',color ='red')  

            sub3.set_ylabel('Power')
            sub3.bar(np.arange(96) ,self.monthlyRemain['3'][:] ,label = 'fixLoad',bottom = self.monthlyHVAC['3'][:] , color ='gray')  
            sub3.bar(np.arange(96) ,self.monthlyHVAC['3'][:] ,label = 'HVAC',color ='red')  

            sub4.set_ylabel('Power')
            sub4.bar(np.arange(96) ,self.monthlyRemain['4'][:] ,label = 'fixLoad',bottom = self.monthlyHVAC['4'][:] , color ='gray')  
            sub4.bar(np.arange(96) ,self.monthlyHVAC['4'][:] ,label = 'HVAC',color ='red')  

            sub5.set_ylabel('Power')
            sub5.bar(np.arange(96) ,self.monthlyRemain['5'][:] ,label = 'fixLoad',bottom = self.monthlyHVAC['5'][:] , color ='gray')  
            sub5.bar(np.arange(96) ,self.monthlyHVAC['5'][:] ,label = 'HVAC',color ='red')  

            sub6.set_ylabel('Power')
            sub6.bar(np.arange(96) ,self.monthlyRemain['6'][:] ,label = 'fixLoad',bottom = self.monthlyHVAC['6'][:] , color ='gray')  
            sub6.bar(np.arange(96) ,self.monthlyHVAC['6'][:] ,label = 'HVAC',color ='red')  

            sub7.set_ylabel('Power')
            sub7.bar(np.arange(96) ,self.monthlyRemain['7'][:] ,label = 'fixLoad',bottom = self.monthlyHVAC['7'][:] , color ='gray')  
            sub7.bar(np.arange(96) ,self.monthlyHVAC['7'][:] ,label = 'HVAC',color ='red')  

            sub8.set_ylabel('Power')
            sub8.bar(np.arange(96) ,self.monthlyRemain['8'][:] ,label = 'fixLoad',bottom = self.monthlyHVAC['8'][:] , color ='gray')  
            sub8.bar(np.arange(96) ,self.monthlyHVAC['8'][:] ,label = 'HVAC',color ='red')  

            sub9.set_ylabel('Power')
            sub9.bar(np.arange(96) ,self.monthlyRemain['9'][:] ,label = 'fixLoad',bottom = self.monthlyHVAC['9'][:] , color ='gray')  
            sub9.bar(np.arange(96) ,self.monthlyHVAC['9'][:] ,label = 'HVAC',color ='red')  

            sub10.set_ylabel('Power')
            sub10.bar(np.arange(96) ,self.monthlyRemain['10'][:] ,label = 'fixLoad',bottom = self.monthlyHVAC['10'][:] , color ='gray')  
            sub10.bar(np.arange(96) ,self.monthlyHVAC['10'][:] ,label = 'HVAC', color ='red')  

            sub11.set_ylabel('Power')
            sub11.bar(np.arange(96) ,self.monthlyRemain['11'][:] ,label = 'fixLoad',bottom = self.monthlyHVAC['11'][:] , color ='gray')  
            sub11.bar(np.arange(96) ,self.monthlyHVAC['11'][:] ,label = 'HVAC', color ='red')  

            sub12.set_ylabel('Power')
            sub12.bar(np.arange(96) ,self.monthlyRemain['12'][:] ,label = 'fixLoad',bottom = self.monthlyHVAC['12'][:] , color ='gray')  
            sub12.bar(np.arange(96) ,self.monthlyHVAC['12'][:] ,label = 'HVAC', color ='red')  

            sub1a = ax1.twinx()
            sub1a.spines['right'].set_position(("axes",1.1))
            sub1a.tick_params(axis='y',colors = "blue")
            sub1a.set_ylabel('price',color='blue')
            sub1a.plot(range(len(self.price)), self.price, label = "price")

            sub2a = ax2.twinx()
            sub2a.spines['right'].set_position(("axes",1.1))
            sub2a.tick_params(axis='y',colors = "blue")
            sub2a.set_ylabel('price',color='blue')
            sub2a.plot(range(len(self.price)), self.price, label = "price")

            sub3a = ax3.twinx()
            sub3a.spines['right'].set_position(("axes",1.1))
            sub3a.tick_params(axis='y',colors = "blue")
            sub3a.set_ylabel('price',color='blue')
            sub3a.plot(range(len(self.price)), self.price, label = "price")

            sub4a = ax4.twinx()
            sub4a.spines['right'].set_position(("axes",1.1))
            sub4a.tick_params(axis='y',colors = "blue")
            sub4a.set_ylabel('price',color='blue')
            sub4a.plot(range(len(self.price)), self.price, label = "price")

            sub5a = ax5.twinx()
            sub5a.spines['right'].set_position(("axes",1.1))
            sub5a.tick_params(axis='y',colors = "blue")
            sub5a.set_ylabel('price',color='blue')
            sub5a.plot(range(len(self.price)), self.price, label = "price")

            sub6a = ax6.twinx()
            sub6a.spines['right'].set_position(("axes",1.1))
            sub6a.tick_params(axis='y',colors = "blue")
            sub6a.set_ylabel('price',color='blue')
            sub6a.plot(range(len(self.price)), self.price, label = "price")

            sub7a = ax7.twinx()
            sub7a.spines['right'].set_position(("axes",1.1))
            sub7a.tick_params(axis='y',colors = "blue")
            sub7a.set_ylabel('price',color='blue')
            sub7a.plot(range(len(self.price)), self.price, label = "price")

            sub8a = ax8.twinx()
            sub8a.spines['right'].set_position(("axes",1.1))
            sub8a.tick_params(axis='y',colors = "blue")
            sub8a.set_ylabel('price',color='blue')
            sub8a.plot(range(len(self.price)), self.price, label = "price")

            sub9a = ax9.twinx()
            sub9a.spines['right'].set_position(("axes",1.1))
            sub9a.tick_params(axis='y',colors = "blue")
            sub9a.set_ylabel('price',color='blue')
            sub9a.plot(range(len(self.price)), self.price, label = "price")

            sub10a = ax10.twinx()
            sub10a.spines['right'].set_position(("axes",1.1))
            sub10a.tick_params(axis='y',colors = "blue")
            sub10a.set_ylabel('price',color='blue')
            sub10a.plot(range(len(self.price)), self.price, label = "price")

            sub11a = ax11.twinx()
            sub11a.spines['right'].set_position(("axes",1.1))
            sub11a.tick_params(axis='y',colors = "blue")
            sub11a.set_ylabel('price',color='blue')
            sub11a.plot(range(len(self.price)), self.price, label = "price")

            sub12a = ax12.twinx()
            sub12a.spines['right'].set_position(("axes",1.1))
            sub12a.tick_params(axis='y',colors = "blue")
            sub12a.set_ylabel('price',color='blue')
            sub12a.plot(range(len(self.price)), self.price, label = "price")


            fig.tight_layout()
            fig.savefig('pic/HVAC/newestHvacResult.png')


    def showNnConfig(self):
        network = self.agent.get_architecture()
        print(network)

    def __del__(self):
        # Close agent and environment
        self.agent.close()
        self.environment.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('please enter the mode: "soc" or "load" or "HVAC" ')
        exit()
    test = Test(sys.argv[1])
    test.main()