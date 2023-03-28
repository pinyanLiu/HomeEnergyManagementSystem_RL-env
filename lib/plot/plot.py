import matplotlib.pyplot as plt
from time import localtime , time,asctime


class Plot():
    def __init__(self,testResult) :
        self.testResult = testResult
        plt.rcParams["figure.figsize"] = (12.8, 9.6)
        self.fig,self.axes = plt.subplots(6,2)
        #ax for power
        self.ax = [self.axes[i,j]for i in range(6) for j in range(2)]
        #sub for price

        #sub2 for temperature , soc

    def remainPower(self):
        for month in range(0,12):
            self.ax[month].set_ylabel('Power')
            self.ax[month].bar(range(96) ,self.testResult[month]['remain'] ,label = 'fixLoad',bottom = self.testResult[month]['deltaSoc'] ,color ='gray') 
            self.ax[month].bar(range(96) ,self.testResult[month]['deltaSoc'],label = 'fixLoad',color ='black') 

    def plotLoadPower(self):
        for month in range(0,12):
            self.ax[month].bar(range(96) ,self.testResult[month]['switch'] ,label = 'switch', color ='green')    

    def plotUnIntLoadPower(self):
        for month in range(0,12):
            self.ax[month].bar(range(96) ,self.testResult[month]['unintSwitch'] ,label = 'unintSwitch', color ='green')    

    def plotIntLoadPower(self):
        for month in range(0,12):
            self.ax[month].bar(range(96) ,self.testResult[month]['switch'] ,label = 'switch', color ='green')    



    
    def price(self):
        self.sub = [sub.twinx() for sub in self.ax]
        for month in range(0,12):
            self.sub[month].set_ylim(0,6.2)
            self.sub[month].set_ylabel('price',color='blue')
            self.sub[month].spines['right'].set_position(("axes",1.1))
            self.sub[month].tick_params(axis='y',colors = 'blue')
            self.sub[month].plot(range(len(self.testResult[month]['price'])),self.testResult[month]['price'], label = "price")  
        

    def soc(self):
        self.sub2 = [sub2.twinx() for sub2 in self.ax]
        for month in range(0,12):
            self.sub2[month].set_ylim(0,1)
            self.sub2[month].set_ylabel('soc',color='red')
            self.sub2[month].spines['right'].set_position(("axes",1.3))
            self.sub2[month].tick_params(axis='y',colors = 'red')
            self.sub2[month].plot(range(96) ,self.testResult[month]['soc'] ,label = 'soc', color ='red')    
                       
    def __pv__(self):
        pass

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
        self.fig.tight_layout()
        self.fig.savefig(dir+str(asctime(localtime(time())))+'.png')

