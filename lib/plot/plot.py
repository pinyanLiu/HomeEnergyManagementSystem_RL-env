import matplotlib.pyplot as plt


class Plot():
    def __init__(self,testResult) :
        self.testResult = testResult
        plt.rcParams["figure.figsize"] = (12.8, 9.6)
        self.fig,self.axes = plt.subplots(6,2)
        self.ax = [self.axes[i,j]for i in range(6) for j in range(2)]
        self.sub = [sub.twinx() for sub in self.ax]
    def power(self):
        for month in range(1,13):
            self.ax[month].set_ylabel('Power')
            self.ax[month].bar(range(96) ,self.testResult[month]['remain'] ,label = 'fixLoad', color ='gray')            
    
    def __price__(self):
        pass

    def __soc__(self):
        pass

    def __pv__(self):
        pass

    def __temperature__(self):
        pass

