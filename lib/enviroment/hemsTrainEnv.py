from  lib.import_data.import_data import ImportData 
from  yaml import load , SafeLoader
from random import randint,uniform
from tensorforce import Environment

class HemsEnv(Environment):
    def __init__(self) :
        '''
        Action space
        observation space
        '''
        #
        # The information of ip should   'NOT'   upload to github
        #
        super().__init__()
        with open("yaml/mysqlData.yaml","r") as f:
            self.mysqlData = load(f,SafeLoader)

        self.host = self.mysqlData['host']
        self.user = self.mysqlData['user']
        self.passwd = self.mysqlData['passwd']
        self.db = self.mysqlData['db']
        self.info = ImportData(host= self.host ,user= self.user ,passwd= self.passwd ,db= self.db)
        #import Base Parameter
        self.BaseParameter = self.info.importBaseParameter()
        #import Grid price
        self.allGridPrice = self.info.importGridPrice()
        self.summerGridPrice = self.allGridPrice['summer_price'].tolist()
        #self.notSummerGridPrice = self.allGridPrice['not_summer_price'].tolist()
        self.notSummerGridPrice = self.allGridPrice['test_price1'].tolist()
        #pick one day from 360 days
        self.i = randint(1,359)
        #import Load 
        self.allLoad = self.info.importTrainingLoad()
        self.Load = self.allLoad.iloc[:,self.i].tolist()
        #import PV
        self.allPV = self.info.importPhotoVoltaic()
        self.state = None
        self.reward = 0
        self.done = False

    def states(self):
        pass

    def actions(self):
        pass


    def close(self):
        pass

    def reset(self):
        '''
        Starting State
        '''
        pass

    def execute(self,actions):
        '''
        interaction of each state(changes while taking action)
        Rewards
        Episode Termination condition
        '''
        pass