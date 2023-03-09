from  lib.import_data.import_data import ImportData 
from  yaml import load , SafeLoader
from random import randint
from tensorforce import Environment

class multiAgentTrainEnv(Environment):
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

        #import all baseParameter  for LLAs
        self.PgridMax = float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='PgridMax']['value'])[0])
        self.batteryCapacity = int(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='batteryCapacity']['value'])[0])
        self.socInit = float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='SOCinit']['value'])[0])
        self.socThreshold = float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='SOCthreshold']['value'])[0])
        self.epsilon = float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='epsilon']['value'])[0])
        self.eta = float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='eta_HVAC']['value'])[0])
        self.A = float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='A(KW/F)']['value'])[0])
        self.max_temperature = float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='max_temperature(F)']['value'])[0])
        self.min_temperature = float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='min_temperature(F)']['value'])[0])
        self.initIndoorTemperature= float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='init_indoor_temperature(F)']['value'])[0])
        self.batteryCapacity=float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='batteryCapacity']['value'])[0])        
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
        #import temperature
        self.allUserSetTemperature = self.info.importUserSetTemperatureF()
        self.allOutdoorTemperature = self.info.importTemperatureF()
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

