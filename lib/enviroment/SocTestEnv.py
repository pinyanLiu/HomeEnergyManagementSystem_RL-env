from lib.enviroment.SocTrainEnv import SocEnv
from gym import make
import numpy as np
from random import randint,uniform
 
class SocTest(SocEnv):
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
        
        self.i += 1
        self.Load = self.allLoad.iloc[:,self.i].tolist()


        #import PV
        if self.i  == 0:
            self.PV = self.allPV['Jan'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif self.i  == 1:
            self.PV = self.allPV['Feb'].tolist()
            self.GridPrice = self.notSummerGridPrice
            # self.GridPrice = self.notSummerGridPrice
        elif self.i  == 2:
            self.PV = self.allPV['Mar'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif self.i  == 3:
            self.PV = self.allPV['Apr'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif self.i  == 4:
            self.PV = self.allPV['May'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif self.i  == 5:
            self.PV = self.allPV['Jun'].tolist()
            self.GridPrice = self.summerGridPrice
        elif self.i  == 6:
            self.PV = self.allPV['July'].tolist()
            self.GridPrice = self.summerGridPrice
        elif self.i  == 7:
            self.PV = self.allPV['Aug'].tolist()
            self.GridPrice = self.summerGridPrice
        elif self.i  == 8:
            self.PV = self.allPV['Sep'].tolist()
            self.GridPrice = self.summerGridPrice
        elif self.i  == 9:
            self.PV = self.allPV['Oct'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif self.i  == 10:
            self.PV = self.allPV['Nov'].tolist()
            self.GridPrice = self.notSummerGridPrice
        elif self.i  == 11:
            self.PV = self.allPV['Dec'].tolist()
            self.GridPrice = self.notSummerGridPrice
        
        remainPower = self.Load[0]-self.PV[0]
        if remainPower < 0:
            chargeMask = [True,True,True,True,True,True,True,True,True,True,True,False,False,False,False,False,False,False,False,False,False]
            dischargeMask = [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True]
            cost = 0
        else : #remain>=0
            chargeMask =  [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True]
            dischargeMask = [True,True,True,True,True,True,True,True,True,True,True,remainPower>1,remainPower>2,remainPower>3,remainPower>4,remainPower>5,remainPower>6,remainPower>7,remainPower>8,remainPower>9,remainPower>10]
        socMask = [1-self.socInit>0.25,1-self.socInit>0.225,1-self.socInit>0.2,1-self.socInit>0.175,1-self.socInit>0.15,1-self.socInit>0.125,1-self.socInit>0.1,1-self.socInit>0.075,1-self.socInit>0.05,1-self.socInit>0.025,True,self.socInit>0.025,self.socInit>0.05,self.socInit>0.075,self.socInit>0.1,self.socInit>0.125,self.socInit>0.15,self.socInit>0.175,self.socInit>0.2,self.socInit>0.225,self.socInit>0.25]
        mask = [a and b for a,b in zip(chargeMask,dischargeMask)]
        mask = [a and b for a,b in zip(mask,socMask)]
        self.action_mask = np.asarray(mask)
        #reset state
        self.state=np.array([0,self.Load[0],self.PV[0],self.socInit,self.GridPrice[0]])
        states = dict(state = self.state,action_mask = self.action_mask)
        
        return states


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
        