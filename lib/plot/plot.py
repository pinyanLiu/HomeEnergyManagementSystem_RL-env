import matplotlib.pyplot as plt
import numpy as np
from time import localtime , time,strftime


class Plot():
    def __init__(self,testResult,single=False) :
        self.testResult = testResult
        if single:
            plt.rcParams["figure.figsize"] = (6.3, 2.4)
            self.fig,self.axes = plt.subplots()
            x = np.arange(96)
            self.axes.set_xticks(x[::5])

        else:
            plt.rcParams["figure.figsize"] = (12.8, 9.6)
            self.fig,self.axes = plt.subplots(6,2)
            self.ax = [self.axes[i,j]for i in range(6) for j in range(2)]

    def remainPower(self,month=False):
        if month != False:
            self.axes.set_ylim(-2,11)
            self.axes.set_ylabel('Power')
            self.axes.bar(range(96) ,self.testResult[month]['remain'] ,label = 'UCpower',color ='gray') 
        else:
            for month in range(0,12):
                self.ax[month].set_ylim(-2,11)
                self.ax[month].set_ylabel('Power')
                self.ax[month].bar(range(96) ,self.testResult[month]['remain'] ,label = 'UCpower',color ='gray') 
 

    def plotUnIntLoadPower(self,id=1,month=False):
        # if id == 1:
        #     color = 'darkslategray'
        # elif id == 2:
        #     color = 'c'
        # else:
        #     color = 'dodgerblue'
        if month != False:
            self.axes.set_ylabel('Power')
            self.axes.bar(range(96) ,self.testResult[month]['unintSwitch'+str(id)] ,label = 'UnintLoadPower', color ='darkslategray')    
            self.axes.legend(loc='lower right', fontsize=6, frameon=True)
        else:
            for month in range(0,12):
                self.ax[month].set_ylabel('Power')
                self.ax[month].bar(range(96) ,self.testResult[month]['unintSwitch'+str(id)] ,label = 'UnintLoadPower', color ='darkslategray')    

    def plotIntLoadPower(self,id=1,month=False):
        # if id == 1:
        #     color = 'lime'
        # elif id == 2:
        #     color = 'seagreen'
        # else:
        #     color = 'aquamarine'
        if month != False:
            self.axes.set_ylabel('Power')
            self.axes.bar(range(96) ,self.testResult[month]['intSwitch'+str(id)] ,label = 'IntLoadPower', color ='lime')   
            self.axes.legend(loc='lower right', fontsize=6, frameon=True)
        else:
            for month in range(0,12):
                self.ax[month].set_ylabel('Power')
                self.ax[month].bar(range(96) ,self.testResult[month]['intSwitch'+str(id)] ,label = 'IntLoadPower', color ='lime')    

    def plotDeltaSOCPower(self,month=False):
        if month != False:
            self.axes.set_ylabel('Power')
            self.axes.bar(range(96) ,self.testResult[month]['deltaSoc']*10 ,label = 'ESSpower', color ='gold')    
            self.axes.legend(loc='lower left', fontsize=6, frameon=True)
        else:
            for month in range(0,12):
                self.ax[month].set_ylabel('Power')
                self.ax[month].bar(range(96) ,self.testResult[month]['deltaSoc']*10 ,label = 'ESSpower', color ='gold')    
            
    def plotPVPower(self,month=False):
        if month != False:
            self.axes.set_ylabel('Power')
            self.axes.bar(range(96) ,-self.testResult[month]['PV'] ,label = 'PVpower', color ='moccasin')  
            self.axes.legend(loc='lower right', fontsize=6, frameon=True)
        else:
            for month in range(0,12):
                self.ax[month].set_ylabel('Power')
                self.ax[month].bar(range(96) ,-self.testResult[month]['PV'] ,label = 'PVpower', color ='moccasin')  

    def plotHVACPower(self,id=1,month=False):
        # if id == 1:
        #     color = 'slateblue'
        # elif id == 2:
        #     color = 'aqua'
        # else:
        #     color = 'dodgerblue'
        if month != False:
            self.axes.set_ylabel('Power')
            self.axes.bar(range(96) ,self.testResult[month]['hvacPower'+str(id)] ,label = 'HVAC_power', color = 'slateblue')  
            self.axes.legend(loc='lower right', fontsize=6, frameon=True)
        else:
            for month in range(0,12):
                self.ax[month].set_ylabel('Power')
                self.ax[month].bar(range(96) ,self.testResult[month]['hvacPower'+str(id)] ,label = 'HVAC_power', color = 'slateblue')  



    def plotPgridMax(self,month=False):
        if month != False:
            self.axes.set_ylabel('Power')
            self.axes.plot(range(96),self.testResult[month]['PgridMax'],linestyle='--',color='red')

        else:
            for month in range(0,12):
                self.ax[month].set_ylabel('Power')
                self.ax[month].plot(range(96),self.testResult[month]['PgridMax'],linestyle='--',color='red')
    
    def price(self,month=False):
        if month != False:
            self.sub = self.axes.twinx()
            self.sub.set_ylim(0,6.5)
            self.sub.set_ylabel('price',color='blue')
            self.sub.spines['right'].set_position(("axes",1))
            self.sub.tick_params(axis='y',colors = 'blue')
            self.sub.plot(range(len(self.testResult[month]['price'])),self.testResult[month]['price'], label = "price")  
            self.sub.legend(loc='upper right', fontsize=6, frameon=True)
        else:
            self.sub = [sub.twinx() for sub in self.ax]
            for month in range(0,12):
                self.sub[month].set_ylim(0,6.5)
                self.sub[month].set_ylabel('price',color='blue')
                self.sub[month].spines['right'].set_position(("axes",1.1))
                self.sub[month].tick_params(axis='y',colors = 'blue')
                self.sub[month].plot(range(len(self.testResult[month]['price'])),self.testResult[month]['price'], label = "price")  
        

    def soc(self,month=False):
        if month != False:
            self.sub5 = self.axes.twinx()
            self.sub5.set_ylim(0,1)
            self.sub5.set_ylabel('soc',color='red')
            self.sub5.spines['right'].set_position(("axes",1.1))
            self.sub5.tick_params(axis='y',colors = 'red')
            self.sub5.plot(range(96) ,self.testResult[month]['soc'] ,label = 'soc', color ='red')    
        else:
            self.sub5 = [sub5.twinx() for sub5 in self.ax]
            for month in range(0,12):
                self.sub5[month].set_ylim(0,1)
                self.sub5[month].set_ylabel('soc',color='red')
                self.sub5[month].spines['right'].set_position(("axes",1.3))
                self.sub5[month].tick_params(axis='y',colors = 'red')
                self.sub5[month].plot(range(96) ,self.testResult[month]['soc'] ,label = 'soc', color ='red')    
                        

    def indoorTemperature(self,id=1,month=False):
        # if id == 1:
        #     color = 'orange'
        # elif id == 2:
        #     color = 'darkorange'
        # else:
        #     color = 'orangered'
        if month != False:
            self.sub2 = self.axes.twinx()
            self.sub2.set_ylim(35,104)
            self.sub2.set_ylabel('Tmp(Fahrenheit)',color='orange')
            self.sub2.spines['right'].set_position(("axes",1.1))
            self.sub2.tick_params(axis='y',colors = 'orange')
            self.sub2.plot(range(96) ,self.testResult[month]['indoorTemperature'+str(id)] ,label = 'indoorTemperature', color = 'orange') 
        else:
            self.sub2 = [sub2.twinx() for sub2 in self.ax]
            for month in range(0,12):
                self.sub2[month].set_ylim(35,104)
                self.sub2[month].set_ylabel('Tmp(Fahrenheit)',color='orange')
                self.sub2[month].spines['right'].set_position(("axes",1.3))
                self.sub2[month].tick_params(axis='y',colors = 'orange')
                self.sub2[month].plot(range(96) ,self.testResult[month]['indoorTemperature'+str(id)] ,label = 'indoorTemperature', color = 'orange') 

    def outdoorTemperature(self,month=False):
        if month != False:
            self.sub2.plot(range(96) ,self.testResult[month]['outdoorTemperature'] ,label = 'outdoorTemperature', color ='goldenrod')
        else:
            for month in range(0,12):
                self.sub2[month].plot(range(96) ,self.testResult[month]['outdoorTemperature'] ,label = 'outdoorTemperature', color ='goldenrod')

    def userSetTemperature(self,id=1,month=False):
        # if id == 1:
        #     color = 'black'
        # elif id == 2:
        #     color = 'dimgray'
        # else:
        #     color = 'lightgray'
        if month != False:
            self.sub2.plot(range(96) ,self.testResult[month]['userSetTemperature'+str(id)] ,label = 'userSetTemperature', color ='black')
            self.sub2.legend(loc='upper left', fontsize=6, frameon=True)
        else:
            for month in range(0,12):
                self.sub2[month].plot(range(96) ,self.testResult[month]['userSetTemperature'+str(id)] ,label = 'userSetTemperature', color ='black')

    def plotReward(self,month=False):
        if month != False:
            self.sub3 = self.axes.twinx()
            self.sub3.plot(range(96) ,self.testResult[month]['reward'] ,label = 'reward', color ='silver')            
        else:
            self.sub3 = [sub3.twinx() for sub3 in self.ax]
            for month in range(0,12):
                self.sub3[month].plot(range(96) ,self.testResult[month]['reward'] ,label = 'reward', color ='silver')            

    def plotResult(self,dir):
        current_time = localtime()
        custom_format = "%Y-%m-%d %H:%M:%S"  # 自定义时间格式
        custom_time = strftime(custom_format, current_time)
        self.fig.tight_layout()
        self.fig.savefig(dir+str(custom_time)+'.png')
        

    def plotIntPreference(self,id=1,month=False):
        # if id == 1:
        #     color = 'lime'
        # elif id == 2:
        #     color = 'seagreen'
        # else:
        #     color = 'aquamarine'
        if month != False:
            self.sub4 = self.axes.twinx() 
            self.sub4.set_ylim(-1.5,4.5)
            self.sub4.set_ylabel('preference',color='lime')
            self.sub4.spines['right'].set_position(("axes",1.1))
            self.sub4.tick_params(axis='y',colors = 'lime')
            self.sub4.plot(range(96) ,self.testResult[month]['intUserPreference'+str(id)] ,label = 'intUserPreference', color ='lime')       
            self.sub4.legend(loc='upper left', fontsize=6, frameon=True)     

        else:
            self.sub4= [sub4.twinx() for sub4 in self.ax]
            for month in range(0,12):
                self.sub4[month].set_ylim(-1.5,4.5)
                self.sub4[month].set_ylabel('preference',color='lime')
                self.sub4[month].spines['right'].set_position(("axes",1.1))
                self.sub4[month].tick_params(axis='y',colors = 'lime')
                self.sub4[month].plot(range(96) ,self.testResult[month]['intUserPreference'+str(id)] ,label = 'intUserPreference', color ='lime')            

    def plotUnintPreference(self,id=1,month=False):
        # if id == 1:
        #     color = 'darkslategray'
        # elif id == 2:
        #     color = 'c'
        # else:
        #     color = 'dodgerblue'
        if month!=False:
            self.sub4 = self.axes.twinx()
            self.sub4.set_ylim(-1.5,4.5)
            self.sub4.set_ylabel('preference',color='darkslategray')
            self.sub4.spines['right'].set_position(("axes",1.1))
            self.sub4.tick_params(axis='y',colors = 'darkslategray')
            self.sub4.plot(range(96) ,self.testResult[month]['unintUserPreference'+str(id)] ,label = 'unintUserPreference', color = 'darkslategray')        
            self.sub4.legend(loc='upper left', fontsize=6, frameon=True)    
        else:
            self.sub4= [sub4.twinx() for sub4 in self.ax]
            for month in range(0,12):
                self.sub4[month].set_ylim(-1.5,4.5)
                self.sub4[month].set_ylabel('preference',color='darkslategray')
                self.sub4[month].spines['right'].set_position(("axes",1.1))
                self.sub4[month].tick_params(axis='y',colors = 'darkslategray')
                self.sub4[month].plot(range(96) ,self.testResult[month]['unintUserPreference'+str(id)] ,label = 'unintUserPreference', color = 'darkslategray')            

