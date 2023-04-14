import matplotlib.pyplot as plt
from time import localtime , time,strftime


class Plot():
    def __init__(self,testResult) :
        self.testResult = testResult
        plt.rcParams["figure.figsize"] = (12.8, 9.6)
        self.fig,self.axes = plt.subplots(6,2)
        #ax for power
        self.ax = [self.axes[i,j]for i in range(6) for j in range(2)]

    def remainPower(self):
        for month in range(0,12):
            self.ax[month].set_ylim(-2,5)
            self.ax[month].set_ylabel('Power')
            self.ax[month].bar(range(96) ,self.testResult[month]['remain'] ,label = 'fixLoad',color ='gray') 
 

    def plotUnIntLoadPower(self):
        for month in range(0,12):
            self.ax[month].set_ylabel('Power')
            self.ax[month].bar(range(96) ,self.testResult[month]['unintSwitch'] ,label = 'unintSwitch', color ='green')    

    def plotIntLoadPower(self):
        for month in range(0,12):
            self.ax[month].set_ylabel('Power')
            self.ax[month].bar(range(96) ,self.testResult[month]['intSwitch'] ,label = 'switch', color ='lime')    

    def plotDeltaSOCPower(self):
        for month in range(0,12):
            self.ax[month].set_ylabel('Power')
            self.ax[month].bar(range(96) ,self.testResult[month]['deltaSoc']*10 ,label = 'power', color ='gold')    
            
    def plotPVPower(self):
        for month in range(0,12):
            self.ax[month].set_ylabel('Power')
            self.ax[month].bar(range(96) ,-self.testResult[month]['PV'] ,label = 'power', color ='moccasin')  

    def plotPgridMax(self):
        for month in range(0,12):
            self.ax[month].set_ylabel('Power')
            self.ax[month].plot(range(96),self.testResult[month]['PgridMax'],linestyle='--',color='red')
    
    def price(self):
        self.sub = [sub.twinx() for sub in self.ax]
        for month in range(0,12):
            self.sub[month].set_ylim(0,6.2)
            self.sub[month].set_ylabel('price',color='blue')
            self.sub[month].spines['right'].set_position(("axes",1.1))
            self.sub[month].tick_params(axis='y',colors = 'blue')
            self.sub[month].plot(range(len(self.testResult[month]['price'])),self.testResult[month]['price'], label = "price")  
        

    def soc(self):
        self.sub5 = [sub5.twinx() for sub5 in self.ax]
        for month in range(0,12):
            self.sub5[month].set_ylim(0,1)
            self.sub5[month].set_ylabel('soc',color='red')
            self.sub5[month].spines['right'].set_position(("axes",1.3))
            self.sub5[month].tick_params(axis='y',colors = 'red')
            self.sub5[month].plot(range(96) ,self.testResult[month]['soc'] ,label = 'soc', color ='red')    
                       

    def indoorTemperature(self):
        self.sub2 = [sub2.twinx() for sub2 in self.ax]
        for month in range(0,12):
            self.sub2[month].set_ylim(35,104)
            self.sub2[month].set_ylabel('Tmp',color='orange')
            self.sub2[month].spines['right'].set_position(("axes",1.3))
            self.sub2[month].tick_params(axis='y',colors = 'orange')
            self.sub2[month].plot(range(96) ,self.testResult[month]['indoorTemperature'] ,label = 'indoorTemperature', color ='orange') 

    def outdoorTemperature(self):
        for month in range(0,12):
            self.sub2[month].plot(range(96) ,self.testResult[month]['outdoorTemperature'] ,label = 'outdoorTemperature', color ='yellow')

    def userSetTemperature(self):
        for month in range(0,12):
            self.sub2[month].plot(range(96) ,self.testResult[month]['userSetTemperature'] ,label = 'userSetTemperature', color ='black')

    def plotReward(self):
        self.sub3 = [sub3.twinx() for sub3 in self.ax]
        for month in range(0,12):
            self.sub3[month].plot(range(96) ,self.testResult[month]['reward'] ,label = 'reward', color ='silver')            

    def plotResult(self,dir):
        current_time = localtime()
        custom_format = "%Y-%m-%d %H:%M:%S"  # 自定义时间格式
        custom_time = strftime(custom_format, current_time)
        self.fig.tight_layout()
        self.fig.savefig(dir+str(custom_time)+'.png')

    def plotOccupancy(self):
        self.sub4= [sub4.twinx() for sub4 in self.ax]
        for month in range(0,12):
            self.sub4[month].set_ylim(0,6)
            self.sub4[month].set_ylabel('OCP',color='pink')
            self.sub4[month].spines['right'].set_position(("axes",1.2))
            self.sub4[month].tick_params(axis='y',colors = 'pink')
            self.sub4[month].plot(range(96) ,self.testResult[month]['occupancy'] ,label = 'Occupancy', color ='pink')            

    def plotIntPreference(self):
        self.sub4= [sub4.twinx() for sub4 in self.ax]
        for month in range(0,12):
            self.sub4[month].set_ylim(-1,4)
            self.sub4[month].set_ylabel('pfr',color='pink')
            self.sub4[month].spines['right'].set_position(("axes",1.2))
            self.sub4[month].tick_params(axis='y',colors = 'pink')
            self.sub4[month].plot(range(96) ,self.testResult[month]['intUserPreference'] ,label = 'intUserPreference', color ='lime')            

    def plotUnintPreference(self):
        self.sub4= [sub4.twinx() for sub4 in self.ax]
        for month in range(0,12):
            self.sub4[month].set_ylim(-1,4)
            self.sub4[month].set_ylabel('pfr',color='pink')
            self.sub4[month].spines['right'].set_position(("axes",1.2))
            self.sub4[month].tick_params(axis='y',colors = 'pink')
            self.sub4[month].plot(range(96) ,self.testResult[month]['unintUserPreference'] ,label = 'unintUserPreference', color ='green')            

