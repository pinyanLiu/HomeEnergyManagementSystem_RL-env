from lib.enviroment.HVACTrainEnv import HvacEnv
import numpy as np

class HvacTest(HvacEnv):
    def __init__(self) :
        super().__init__()

        #each month pick one day for testing
        self.i = 0
    #import Load 
        self.allLoad = self.info.importTestingLoad()
        self.Load = self.allLoad.iloc[:,self.i].tolist()

    def states(self):
        return super().states()

    def actions(self):
        return super().actions()

    def execute(self, actions):
        return super().execute(actions)

    def close(self):
        return super().close()


    def reset(self):
        '''
        Starting State
        '''
        #each month pick one day for testing
        self.i += 1
        self.Load = self.allLoad.iloc[:,self.i].to_list()

    #setting PV
        if self.i == 0:
            self.outdoorTemperature = self.allOutdoorTemperature['Jan'].tolist()
            self.userSetTemperature = self.allUserSetTemperature['Jan'].tolist()
            self.PV = self.allPV['Jan'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif self.i == 1:
            self.outdoorTemperature = self.allOutdoorTemperature['Feb'].tolist()
            self.userSetTemperature = self.allUserSetTemperature['Feb'].tolist()
            self.PV = self.allPV['Feb'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif self.i == 2:
            self.outdoorTemperature = self.allOutdoorTemperature['Mar'].tolist()
            self.userSetTemperature = self.allUserSetTemperature['Mar'].tolist()
            self.PV = self.allPV['Mar'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif self.i == 3:
            self.outdoorTemperature = self.allOutdoorTemperature['Apr'].tolist()
            self.userSetTemperature = self.allUserSetTemperature['Apr'].tolist()
            self.PV = self.allPV['Apr'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif self.i == 4:
            self.outdoorTemperature = self.allOutdoorTemperature['May'].tolist()
            self.userSetTemperature = self.allUserSetTemperature['May'].tolist()
            self.PV = self.allPV['May'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif self.i == 5:
            self.outdoorTemperature = self.allOutdoorTemperature['Jun'].tolist()
            self.userSetTemperature = self.allUserSetTemperature['Jun'].tolist()
            self.PV = self.allPV['Jun'].tolist()
            self.GridPrice = self.summerGridPrice
        elif self.i == 6:
            self.outdoorTemperature = self.allOutdoorTemperature['July'].tolist()
            self.userSetTemperature = self.allUserSetTemperature['July'].tolist()
            self.PV = self.allPV['July'].tolist()
            self.GridPrice = self.summerGridPrice
        elif self.i == 7:
            self.outdoorTemperature = self.allOutdoorTemperature['Aug'].tolist()
            self.userSetTemperature = self.allUserSetTemperature['Aug'].tolist()
            self.PV = self.allPV['Aug'].tolist()
            self.GridPrice = self.summerGridPrice
        elif self.i == 8:
            self.outdoorTemperature = self.allOutdoorTemperature['Sep'].tolist()
            self.userSetTemperature = self.allUserSetTemperature['Sep'].tolist()
            self.PV = self.allPV['Sep'].tolist()
            self.GridPrice = self.summerGridPrice
        elif self.i == 9:
            self.outdoorTemperature = self.allOutdoorTemperature['Oct'].tolist()
            self.userSetTemperature = self.allUserSetTemperature['Oct'].tolist()
            self.PV = self.allPV['Oct'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif self.i == 10:
            self.outdoorTemperature = self.allOutdoorTemperature['Nov'].tolist()
            self.userSetTemperature = self.allUserSetTemperature['Nov'].tolist()
            self.PV = self.allPV['Nov'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif self.i == 11:
            self.outdoorTemperature = self.allOutdoorTemperature['Dcb'].tolist()
            self.userSetTemperature = self.allUserSetTemperature['Dcb'].tolist()
            self.PV = self.allPV['Dec'].tolist()
            self.GridPrice = self.notSummerGridPrice

        #reset state
        self.state=np.array([0,self.Load[0],self.PV[0],self.GridPrice[0],self.deltaSoc[0],self.initIndoorTemperature,self.outdoorTemperature[0],self.userSetTemperature[0]])
        return self.state

