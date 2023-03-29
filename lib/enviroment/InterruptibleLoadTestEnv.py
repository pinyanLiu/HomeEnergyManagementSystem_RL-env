from lib.enviroment.InterruptibleLoadTrainEnv import IntEnv
from gym import make
import numpy as np
from lib.loads.interrupted import AC

class IntTest(IntEnv):
    def __init__(self) :
        super().__init__()
        #import Base Parameter
        self.intload_demand =  int(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='intload_demand']['value'])[0])

        self.intload_power =  float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='intload_power']['value'])[0])

        #each month pick one day for testing
        self.i = 0
        #import Load 
        self.allLoad = self.info.importTestingLoad()
        self.Load = self.allLoad.iloc[:,self.i].tolist()
        #set AC parameter
        self.interruptibleLoad = AC(demand=self.intload_demand,AvgPowerConsume=self.intload_power)

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
        self.interruptibleLoad.reset()
        self.i += 1
        self.Load = self.allLoad.iloc[:,self.i].tolist()

        #import PV
        if self.i  == 0:
            self.PV = self.allPV['Jan'].tolist()
            self.GridPrice = self.notSummerGridPrice
            self.deltaSOC = self.allDeltaSOC['Jan'].tolist()
            self.userPreference = self.allUserPreference['1'].tolist()
        elif self.i  == 1:
            self.PV = self.allPV['Feb'].tolist()
            self.GridPrice = self.notSummerGridPrice
            self.deltaSOC = self.allDeltaSOC['Feb'].tolist()
            self.userPreference = self.allUserPreference['2'].tolist()

        elif self.i  == 2:
            self.PV = self.allPV['Mar'].tolist()
            self.GridPrice = self.notSummerGridPrice
            self.deltaSOC = self.allDeltaSOC['Mar'].tolist()
            self.userPreference = self.allUserPreference['3'].tolist()
        elif self.i  == 3:
            self.PV = self.allPV['Apr'].tolist()
            self.GridPrice = self.notSummerGridPrice
            self.deltaSOC = self.allDeltaSOC['Apr'].tolist()
            self.userPreference = self.allUserPreference['4'].tolist()

        elif self.i  == 4:
            self.PV = self.allPV['May'].tolist()
            self.GridPrice = self.notSummerGridPrice
            self.deltaSOC = self.allDeltaSOC['May'].tolist()
            self.userPreference = self.allUserPreference['5'].tolist()
        elif self.i  == 5:
            self.PV = self.allPV['Jun'].tolist()
            self.GridPrice = self.summerGridPrice
            self.deltaSOC = self.allDeltaSOC['Jun'].tolist()
            self.userPreference = self.allUserPreference['6'].tolist()

        elif self.i  == 6:
            self.PV = self.allPV['July'].tolist()
            self.GridPrice = self.summerGridPrice
            self.deltaSOC = self.allDeltaSOC['July'].tolist()
            self.userPreference = self.allUserPreference['7'].tolist()
        elif self.i  == 7:
            self.PV = self.allPV['Aug'].tolist()
            self.GridPrice = self.summerGridPrice
            self.deltaSOC = self.allDeltaSOC['Aug'].tolist()
            self.userPreference = self.allUserPreference['8'].tolist()
        elif self.i  == 8:
            self.PV = self.allPV['Sep'].tolist()
            self.GridPrice = self.summerGridPrice
            self.deltaSOC = self.allDeltaSOC['Sep'].tolist()
            self.userPreference = self.allUserPreference['9'].tolist()
        elif self.i  == 9:
            self.PV = self.allPV['Oct'].tolist()
            self.GridPrice = self.notSummerGridPrice
            self.deltaSOC = self.allDeltaSOC['Oct'].tolist()
            self.userPreference = self.allUserPreference['10'].tolist()
        elif self.i  == 10:
            self.PV = self.allPV['Nov'].tolist()
            self.GridPrice = self.notSummerGridPrice
            self.deltaSOC = self.allDeltaSOC['Nov'].tolist()
            self.userPreference = self.allUserPreference['11'].tolist()

        elif self.i  == 11:
            self.PV = self.allPV['Dec'].tolist()
            self.GridPrice = self.notSummerGridPrice
            self.deltaSOC = self.allDeltaSOC['Dcb'].tolist()
            self.userPreference = self.allUserPreference['12'].tolist()

        #reset state
        self.state=np.array([0,self.Load[0],self.PV[0],self.GridPrice[0],self.deltaSOC[0],self.interruptibleLoad.demand,self.userPreference[0]])
        #action mask
        PgridMaxExceed = (self.Load[0]+self.deltaSOC[0]+self.interruptibleLoad.AvgPowerConsume-self.PV[0]) >= self.PgridMax

        self.action_mask = np.asarray([True,self.state[5]>0 and not PgridMaxExceed])
        return self.state


if __name__ == '__main__':
    env = make("Hems-v1")
#     # Initialize episode
    states = env.reset()
    done = False
    step = 0
    Totalreward = 0
    while not done: # Episode timestep
        actions = env.action_space.sample()
        states, reward, done , info = env.step(action=actions)
        Totalreward += reward
    print(states)
        