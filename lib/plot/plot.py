import matplotlib.pyplot as plt


class Plot():
    def __init__(self,testResult) :
        self.testResult = testResult
        plt.rcParams["figure.figsize"] = (12.8, 9.6)
        self.fig,self.axes = plt.subplots(6,2)
        self.ax = [self.axes[i,j]for i in range(6) for j in range(2)]
        self.sub = [sub.twinx() for sub in self.ax]
    def power(self):
        for month in range(0,12):
            self.ax[month].set_ylabel('Power')
            if self.testResult[month]['switch'].tolist():
                self.ax[month].bar(range(96) ,self.testResult[month]['remain'] ,label = 'fixLoad',bottom = self.testResult[month]['switch'] ,color ='gray') 
            else:
                self.ax[month].bar(range(96) ,self.testResult[month]['remain'] ,label = 'fixLoad', color ='gray') 

    def plotUninterruptible(self):
        for month in range(0,12):
            self.ax[month].bar(range(96) ,self.testResult[month]['switch'] ,label = 'switch', color ='green')            

    
    def price(self):
        for month in range(0,12):
            self.sub[month].set_ylabel('price',color='blue')
            self.sub[month].spines['right'].set_position(("axes",1.1))
            self.sub[month].tick_params(axis='y',colors = 'blue')
            self.sub[month].plot(range(len(self.testResult[month]['price'])),self.testResult[month]['price'], label = "price")  
        

    def __soc__(self):
        for month in range(0,12):
            self.ax[month].set_ylabel('Power')
            self.ax[month].bar(range(96) ,self.testResult[month]['remain'] ,label = 'fixLoad', color ='gray')            

    def __pv__(self):
        pass

    def __temperature__(self):
        pass

