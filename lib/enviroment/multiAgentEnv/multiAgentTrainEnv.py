from  lib.import_data.import_data import ImportData 
from  yaml import load , SafeLoader
from tensorforce import Environment,Agent
from lib.enviroment.multiAgentEnv.LLA.LLA import socLLA,hvacLLA,intLLA,unintLLA 

import numpy as np
from  gym import spaces
from random import randint,uniform
from lib.loads.interrupted import AC
from lib.loads.uninterrupted import WM

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
        self.notSummerGridPrice = self.allGridPrice['not_summer_price'].tolist()
        #self.notSummerGridPrice = self.allGridPrice['summer_price'].tolist()
        self.intload_demand =  int(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='intload_demand']['value'])[0])
        self.intload_power =  float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='intload_power']['value'])[0])
        self.unload_demand =  int(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='unload_demand']['value'])[0])
        self.unload_period =  int(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='unload_period']['value'])[0])
        self.unload_power =  float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='unload_power']['value'])[0])

        #pick one day from training data
        self.i = randint(1,359)

        #import Load 
        self.allLoad = self.info.importTrainingLoad()
        self.Load = self.allLoad.iloc[:,self.i].tolist()
        #import PV
        self.allPV = self.info.importPhotoVoltaic()
        #import temperature
        self.allUserSetTemperature1 = self.info.importUserSetTemperatureF()
        self.allUserSetTemperature2 = self.info.importUserSetTemperatureF2()
        self.allUserSetTemperature3 = self.info.importUserSetTemperatureF3()
        self.allOutdoorTemperature = self.info.importTemperatureF()
        self.allStatisticalData = self.info.importStatisticalData()

        #unint  / int object 
        self.interruptibleLoad = AC(demand=self.intload_demand,AvgPowerConsume=self.intload_power)
        self.uninterruptibleLoad = WM(demand=self.unload_demand,executePeriod=self.unload_period,AvgPowerConsume=self.unload_power)

        # self.interruptibleLoad = AC(demand=randint(1,49),AvgPowerConsume=1.5)
        # self.uninterruptibleLoad = WM(demand=randint(3,24),executePeriod=3,AvgPowerConsume=uniform(0.5,1))
        self.allIntPreference = self.info.importIntPreference()
        self.allUnintPreference = self.info.importUnIntPreference()



        #construct all LLAs
        
        self.socAgent = socLLA(mean=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='SOC']['mean'])[0]),std=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='SOC']['std'])[0]),baseParameter=self.BaseParameter)

        
        self.hvacAgent1 = hvacLLA(mean=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='HVAC']['mean'])[0]),std=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='HVAC']['std'])[0]),baseParameter=self.BaseParameter,allOutdoorTemperature=self.allOutdoorTemperature,allUserSetTemperature=self.allUserSetTemperature1)

        self.hvacAgent2 = hvacLLA(mean=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='HVAC']['mean'])[0]),std=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='HVAC']['std'])[0]),baseParameter=self.BaseParameter,allOutdoorTemperature=self.allOutdoorTemperature,allUserSetTemperature=self.allUserSetTemperature2)

        self.hvacAgent3 = hvacLLA(mean=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='HVAC']['mean'])[0]),std=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='HVAC']['std'])[0]),baseParameter=self.BaseParameter,allOutdoorTemperature=self.allOutdoorTemperature,allUserSetTemperature=self.allUserSetTemperature3)
        
        
        self.intAgent = intLLA(mean=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='Interruptible']['mean'])[0]),std=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='Interruptible']['std'])[0]),baseParameter=self.BaseParameter,Int=self.interruptibleLoad)
        
        
        self.unIntAgent = unintLLA(mean=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='Uninterruptible']['mean'])[0]),std=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='Uninterruptible']['std'])[0]),baseParameter=self.BaseParameter,unInt=self.uninterruptibleLoad)


        self.state = None
        self.totalState = {
            "sampleTime":0,
            "fixLoad":0,
            "PV":0,
            "SOC":0,
            "pricePerHour":0,
            "deltaSoc":0,
            "indoorTemperature1":0,
            "indoorTemperature2":0,
            "indoorTemperature3":0,
            "outdoorTemperature":0,
            "userSetTemperature1":0,
            "userSetTemperature2":0,
            "userSetTemperature3":0,
            "hvacPower1":0,
            "hvacPower2":0,
            "hvacPower3":0,
            "intSwitch":0,
            "intRemain":0,
            "intPreference":0,
            "unintRemain":0,
            "unintSwitch":0,
            "unintPreference":0,
            "order":0,
            "PgridMax":0
        }
        self.reward = 0
        self.done = False
        self.action_mask = [True,True,True,True,True,True,True]
        self.interruptibleLoadActionMask = [True,True]
        self.uninterruptibleLoadActionMask = [True,True]
        self.tempActionMaskFlag = False # use for recording whether remain>pgridmax in last step


    def states(self):
        #observation space 
        #state abstraction
        upperLimit = np.array(
            [
                #time block
                95,
                #SOC
                1.0,
                #Remain
                12.0,
                #price per hour
                6.2,
                #HVAC1 state
                1,
                #HVAC2 state
                1,
                #HVAC3 state
                1,
                #Int state
                1,
                #Unint state
                1,
                #Int preference
                4,
                #unInt preference
                4,
                #Order
                5
            ],
            dtype=np.float32,
        )
        lowerLimit = np.array(
            [
                #time block
                0,
                #SOC
                0.0,
                #Remain
                -10.0,
                #pricePerHour
                0.0,
                #HVAC1 state
                -1,
                #HVAC2 state
                -1,
                #HVAC3 state
                -1,
                #Int state
                0,
                #Unint state
                0,
                #Int preference
                -1,
                #unInt preference
                -1,
                #Order
                0
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(lowerLimit,upperLimit,dtype=np.float32)
        return dict(type='float',shape=self.observation_space.shape,min_value=lowerLimit,max_value=upperLimit)

    def actions(self):
        return dict(type='int',num_values=7)

    def close(self):
        return super().close()

    def reset(self):
        '''
        Starting State
        '''
        #pick one day from 360 days
        self.i = randint(1,359)
        #AC/WM object
        self.interruptibleLoad = AC(demand=self.intload_demand,AvgPowerConsume=self.intload_power)
        self.uninterruptibleLoad = WM(demand=self.unload_demand,executePeriod=self.unload_period,AvgPowerConsume=self.unload_power)
        self.randomDeltaPrice  = [uniform(-1,1) for _ in range(96)]
        self.randomDeltaPV = [uniform(-0.5,0.5) for _ in range(96)]        
        self.randomTemperature = [uniform(-2,2)for _ in range(96)]
        self.randomDeltaPreference = [randint(-1,1) for _ in range(96)]

        #import PV,outTmp,userTmp,GridPrice
        if int( self.i / 30) == 0:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['Jan'].tolist(),self.randomTemperature)]
            self.userSetTemperature1 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature1['Jan'].tolist(),self.randomTemperature)]
            self.userSetTemperature2 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature2['Jan'].tolist(),self.randomTemperature)]
            self.userSetTemperature3 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature3['Jan'].tolist(),self.randomTemperature)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Jan'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]  
            self.intUserPreference = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference['1'].tolist(),self.randomDeltaPreference)]
            self.unintPreference = [min(max(x+y,-1),4) for x,y in zip(self.allUnintPreference['1'].tolist(),self.randomDeltaPreference)]

            
        elif int(self.i / 30) == 1:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['Feb'].tolist(),self.randomTemperature)]
            self.userSetTemperature1 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature1['Feb'].tolist(),self.randomTemperature)]
            self.userSetTemperature2 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature2['Feb'].tolist(),self.randomTemperature)]
            self.userSetTemperature3 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature3['Feb'].tolist(),self.randomTemperature)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Feb'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
            self.intUserPreference = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference['2'].tolist(),self.randomDeltaPreference)]
            self.unintPreference = [min(max(x+y,-1),4) for x,y in zip(self.allUnintPreference['2'].tolist(),self.randomDeltaPreference)]

        elif int(self.i / 30) == 2:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['Mar'].tolist(),self.randomTemperature)]
            self.userSetTemperature1 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature1['Mar'].tolist(),self.randomTemperature)]
            self.userSetTemperature2 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature2['Mar'].tolist(),self.randomTemperature)]
            self.userSetTemperature3 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature3['Mar'].tolist(),self.randomTemperature)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Mar'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
            self.intUserPreference = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference['3'].tolist(),self.randomDeltaPreference)]
            self.unintPreference = [min(max(x+y,-1),4) for x,y in zip(self.allUnintPreference['3'].tolist(),self.randomDeltaPreference)]

        elif int(self.i / 30) == 3:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['Apr'].tolist(),self.randomTemperature)]
            self.userSetTemperature1 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature1['Apr'].tolist(),self.randomTemperature)]
            self.userSetTemperature2 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature2['Apr'].tolist(),self.randomTemperature)]
            self.userSetTemperature3 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature3['Apr'].tolist(),self.randomTemperature)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Apr'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
            self.intUserPreference = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference['4'].tolist(),self.randomDeltaPreference)]
            self.unintPreference = [min(max(x+y,-1),4) for x,y in zip(self.allUnintPreference['4'].tolist(),self.randomDeltaPreference)]


        elif int(self.i / 30) == 4:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['May'].tolist(),self.randomTemperature)]
            self.userSetTemperature1 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature1['May'].tolist(),self.randomTemperature)]
            self.userSetTemperature2 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature2['May'].tolist(),self.randomTemperature)]
            self.userSetTemperature3 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature3['May'].tolist(),self.randomTemperature)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['May'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
            self.intUserPreference = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference['5'].tolist(),self.randomDeltaPreference)]
            self.unintPreference = [min(max(x+y,-1),4) for x,y in zip(self.allUnintPreference['5'].tolist(),self.randomDeltaPreference)]

        elif int(self.i / 30) == 5:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['Jun'].tolist(),self.randomTemperature)]
            self.userSetTemperature1 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature1['Jun'].tolist(),self.randomTemperature)]
            self.userSetTemperature2 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature2['Jun'].tolist(),self.randomTemperature)]
            self.userSetTemperature3 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature3['Jun'].tolist(),self.randomTemperature)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Jun'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
            self.intUserPreference = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference['6'].tolist(),self.randomDeltaPreference)]
            self.unintPreference = [min(max(x+y,-1),4) for x,y in zip(self.allUnintPreference['6'].tolist(),self.randomDeltaPreference)]

        elif int(self.i / 30) == 6:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['July'].tolist(),self.randomTemperature)]
            self.userSetTemperature1 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature1['July'].tolist(),self.randomTemperature)]
            self.userSetTemperature2 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature2['July'].tolist(),self.randomTemperature)]
            self.userSetTemperature3 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature3['July'].tolist(),self.randomTemperature)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['July'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.summerGridPrice,self.randomDeltaPrice) ]
            self.intUserPreference = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference['7'].tolist(),self.randomDeltaPreference)]
            self.unintPreference = [min(max(x+y,-1),4) for x,y in zip(self.allUnintPreference['7'].tolist(),self.randomDeltaPreference)]

        elif int(self.i / 30) == 7:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['Aug'].tolist(),self.randomTemperature)]
            self.userSetTemperature1 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature1['Aug'].tolist(),self.randomTemperature)]
            self.userSetTemperature2 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature2['Aug'].tolist(),self.randomTemperature)]
            self.userSetTemperature3 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature3['Aug'].tolist(),self.randomTemperature)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Aug'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.summerGridPrice,self.randomDeltaPrice) ]
            self.intUserPreference = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference['8'].tolist(),self.randomDeltaPreference)]
            self.unintPreference = [min(max(x+y,-1),4) for x,y in zip(self.allUnintPreference['8'].tolist(),self.randomDeltaPreference)]

        elif int(self.i / 30) == 8:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['Sep'].tolist(),self.randomTemperature)]
            self.userSetTemperature1 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature1['Sep'].tolist(),self.randomTemperature)]
            self.userSetTemperature2 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature2['Sep'].tolist(),self.randomTemperature)]
            self.userSetTemperature3 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature3['Sep'].tolist(),self.randomTemperature)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Sep'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.summerGridPrice,self.randomDeltaPrice) ]
            self.intUserPreference = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference['9'].tolist(),self.randomDeltaPreference)]
            self.unintPreference = [min(max(x+y,-1),4) for x,y in zip(self.allUnintPreference['9'].tolist(),self.randomDeltaPreference)]

        elif int(self.i / 30) == 9:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['Oct'].tolist(),self.randomTemperature)]
            self.userSetTemperature1 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature1['Oct'].tolist(),self.randomTemperature)]
            self.userSetTemperature2 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature2['Oct'].tolist(),self.randomTemperature)]
            self.userSetTemperature3 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature3['Oct'].tolist(),self.randomTemperature)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Oct'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.summerGridPrice,self.randomDeltaPrice) ]
            self.intUserPreference = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference['10'].tolist(),self.randomDeltaPreference)]
            self.unintPreference = [min(max(x+y,-1),4) for x,y in zip(self.allUnintPreference['10'].tolist(),self.randomDeltaPreference)]

        elif int(self.i / 30) == 10:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['Nov'].tolist(),self.randomTemperature)]
            self.userSetTemperature1 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature1['Nov'].tolist(),self.randomTemperature)]
            self.userSetTemperature2 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature2['Nov'].tolist(),self.randomTemperature)]
            self.userSetTemperature3 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature3['Nov'].tolist(),self.randomTemperature)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Nov'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
            self.intUserPreference = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference['11'].tolist(),self.randomDeltaPreference)]
            self.unintPreference = [min(max(x+y,-1),4) for x,y in zip(self.allUnintPreference['11'].tolist(),self.randomDeltaPreference)]

        elif int(self.i / 30) == 11:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['Dcb'].tolist(),self.randomTemperature)]
            self.userSetTemperature1 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature1['Dcb'].tolist(),self.randomTemperature)]
            self.userSetTemperature2 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature2['Dcb'].tolist(),self.randomTemperature)]
            self.userSetTemperature3 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature3['Dcb'].tolist(),self.randomTemperature)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Dec'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
            self.intUserPreference = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference['12'].tolist(),self.randomDeltaPreference)]
            self.unintPreference = [min(max(x+y,-1),4) for x,y in zip(self.allUnintPreference['12'].tolist(),self.randomDeltaPreference)]



        #reset state
        self.totalState = {
            "sampleTime":0,
            "fixLoad":self.Load[0],
            "PV":self.PV[0],
            "SOC":self.socInit,
            "pricePerHour":self.GridPrice[0],
            "deltaSoc":0,
            "indoorTemperature1":self.initIndoorTemperature,
            "indoorTemperature2":self.initIndoorTemperature,
            "indoorTemperature3":self.initIndoorTemperature,
            "outdoorTemperature":self.outdoorTemperature[0],
            "userSetTemperature1":self.userSetTemperature1[0],
            "userSetTemperature2":self.userSetTemperature2[0],
            "userSetTemperature3":self.userSetTemperature3[0],
            "hvacPower1":0,
            "hvacPower2":0,
            "hvacPower3":0,
            "intRemain":self.interruptibleLoad.demand,
            "intSwitch":self.interruptibleLoad.switch,
            "intPreference":self.intUserPreference[0],
            "unintRemain":self.uninterruptibleLoad.demand*self.uninterruptibleLoad.executePeriod,
            "unintSwitch":self.uninterruptibleLoad.switch,
            "unintPreference":self.unintPreference[0],
            "order":0,
            "PgridMax":self.PgridMax
        }
        self.interruptibleLoadActionMask = [True,True]
        self.uninterruptibleLoadActionMask = [True,True]
        self.action_mask = [True,True,True,True,True,True,True]
        self.state = self.stateAbstraction(self.totalState)
        self.socAgent.agent.internals = self.socAgent.agent.initial_internals()
        self.hvacAgent1.agent.internals = self.hvacAgent1.agent.initial_internals()
        self.hvacAgent2.agent.internals = self.hvacAgent2.agent.initial_internals()
        self.hvacAgent3.agent.internals = self.hvacAgent3.agent.initial_internals()
        self.intAgent.agent.internals = self.intAgent.agent.initial_internals()
        self.unIntAgent.agent.internals = self.unIntAgent.agent.initial_internals()
        self.socAgent.environment.reset()
        self.hvacAgent1.environment.reset()
        self.hvacAgent2.environment.reset()
        self.hvacAgent3.environment.reset()
        self.intAgent.environment.reset()
        self.unIntAgent.environment.reset()


        return self.state


    def execute(self,actions):
        '''
        interaction of each state(changes while taking action)
        Rewards
        Episode Termination condition
        '''
        reward = []
        sampleTime,soc,remain,pricePreHour,hvacState1,hvacState2,hvacState3,intState,unIntState,intPreference,unintPreference,order = self.state
#         #print(self.totalState)

        #choose one load executing in this sampleTime order .Load which has been execute in the same sampleTime would be auto block by action mask

        #soc
        if actions == 0 :
            # print('soc')
            self.socAgent.getState(self.totalState)
            self.socAgent.environment.updateState(self.socAgent.states)
            self.socAgent.execute()
            # self.socAgent.rewardStandardization()
            # reward.append(self.socAgent.reward)
            self.updateTotalState("soc")
            self.action_mask = [a and b for a,b in zip(self.action_mask , [False,True,True,True,True,True,True])]
        #hvac1
        elif actions == 1:
            # print('hvac')
            self.hvacAgent1.getState(self.totalState)
            self.hvacAgent1.environment.updateState(self.hvacAgent1.states)
            self.hvacAgent1.execute()
            # self.hvacAgent1.rewardStandardization()
            # # reward.append(self.hvacAgent1.reward)            
            self.updateTotalState("hvac1")
            self.action_mask = [a and b for a,b in zip(self.action_mask , [True,False,True,True,True,True,True])]
        #hvac2
        elif actions == 2:
            # print('hvac')
            self.hvacAgent2.getState(self.totalState)
            self.hvacAgent2.environment.updateState(self.hvacAgent2.states)
            self.hvacAgent2.execute()
            # self.hvacAgent2.rewardStandardization()
            # # reward.append(self.hvacAgent2.reward)            
            self.updateTotalState("hvac2")
            self.action_mask = [a and b for a,b in zip(self.action_mask , [True,True,False,True,True,True,True])]
        #hvac3
        elif actions == 3:
            # print('hvac')
            self.hvacAgent3.getState(self.totalState)
            self.hvacAgent3.environment.updateState(self.hvacAgent3.states)
            self.hvacAgent3.execute()
            # self.hvacAgent3.rewardStandardization()
            # # reward.append(self.hvacAgent3.reward)            
            self.updateTotalState("hvac3")
            self.action_mask = [a and b for a,b in zip(self.action_mask , [True,True,True,False,True,True,True])]
        
        #int
        elif actions == 4:
            # print('int')
            self.intAgent.getState(self.totalState,self.interruptibleLoadActionMask)
            self.intAgent.environment.updateState(self.intAgent.states,self.interruptibleLoad)
            self.intAgent.execute()
            # self.intAgent.rewardStandardization()
            # # reward.append(self.intAgent.reward)            
            self.updateTotalState("int")
            self.action_mask = [a and b for a,b in zip(self.action_mask , [True,True,True,True,False,True,True])]
        #unint
        elif actions == 5:
            # print('unint')
            self.unIntAgent.getState(self.totalState,self.uninterruptibleLoadActionMask)
            self.unIntAgent.environment.updateState(self.unIntAgent.states,self.uninterruptibleLoad)
            self.unIntAgent.execute()
            # self.unIntAgent.rewardStandardization()
            # # reward.append(self.unIntAgent.reward)
            self.updateTotalState("unint")
            self.action_mask = [a and b for a,b in zip(self.action_mask , [True,True,True,True,True,False,True])]
        #none
        else:
            # print('none')
            self.updateTotalState("None")
            reward.append(-3)
        # print(self.action_mask,order)


        # Pgrid Max action mask
        # tempActionMask is used as a temp list to record the current action mask.
        # tempActionMaskFlag is used for recording whether previous step exceeded Pgrid Max limit.
        # if remain exceeded the Pgrid Max limit, Flag = True, and the HLA can only choose SOC or no-action
        # if the Flag = True, current action mask have to restore the previous action mask (tempActionMask[1:])
        self.state = self.stateAbstraction(self.totalState)
        if self.tempActionMaskFlag :
            self.tempActionMaskFlag = False
            self.action_mask[1:] = self.tempActionMask[1:]

        if self.state[2]>self.PgridMax:
            self.tempActionMask = self.action_mask
            self.tempActionMaskFlag = True
            self.action_mask = [a and b for a,b in zip(self.action_mask , [True,False,False,False,False,False,True])]
        

        if order == 5:
            if self.action_mask[1] == True:
                self.totalState["indoorTemperature1"] = self.epsilon*self.totalState["indoorTemperature1"]+(1-self.epsilon)*(self.totalState["outdoorTemperature"])
            if self.action_mask[2] == True:
                self.totalState["indoorTemperature2"] = self.epsilon*self.totalState["indoorTemperature2"]+(1-self.epsilon)*(self.totalState["outdoorTemperature"])
            if self.action_mask[3] == True:
                self.totalState["indoorTemperature3"] = self.epsilon*self.totalState["indoorTemperature3"]+(1-self.epsilon)*(self.totalState["outdoorTemperature"])
            if self.action_mask[5]==True:
                self.unIntAgent.environment.uninterruptibleLoad.step()
            self.action_mask = [True,True,True,True,True,True,True]
            reward.append(hvacState1)
            reward.append(hvacState2)
            reward.append(hvacState3)
            reward.append(2*intState*intPreference/self.interruptibleLoad.demand)
            reward.append(2*unIntState*unintPreference/(self.uninterruptibleLoad.demand*self.uninterruptibleLoad.executePeriod))

            if(self.state[2]>self.PgridMax):  
                # print("PGRID MAX OVER!!!")
                reward.append(170*(self.PgridMax-self.state[2]))

        #check if all day is done
        done =  bool(sampleTime == 95 and order == 5)

        reward = sum(reward)/6
        states = dict(state=self.state,action_mask = self.action_mask)


        return states, done ,reward


    def updateTotalState(self,mode) :
        if mode == "soc":
            self.totalState["deltaSoc"] = self.socAgent.actions[0]
            self.totalState["SOC"]+=self.socAgent.actions[0]
            if self.totalState["SOC"]>=1 :
                self.totalState["SOC"]=1
            elif self.totalState["SOC"]<=0:
                self.totalState["SOC"]=0

        elif mode == "hvac1":
            self.totalState["fixLoad"]+=self.hvacAgent1.actions[0]
            self.totalState["indoorTemperature1"] = self.hvacAgent1.states["state"][5]
            self.totalState["hvacPower1"] = self.hvacAgent1.actions[0]

        elif mode == "hvac2":
            self.totalState["fixLoad"]+=self.hvacAgent2.actions[0]
            self.totalState["indoorTemperature2"] = self.hvacAgent2.states["state"][5]
            self.totalState["hvacPower2"] = self.hvacAgent2.actions[0]

        elif mode == "hvac3":
            self.totalState["fixLoad"]+=self.hvacAgent3.actions[0]
            self.totalState["indoorTemperature3"] = self.hvacAgent3.states["state"][5]
            self.totalState["hvacPower3"] = self.hvacAgent3.actions[0]

        elif mode == "int":
            self.interruptibleLoad = self.intAgent.environment.interruptibleLoad
            self.interruptibleLoadActionMask = self.intAgent.states["action_mask"]
            self.totalState["intRemain"] = self.intAgent.states["state"][5]
            self.totalState["intSwitch"] = self.intAgent.actions
            if self.intAgent.actions==1:
                self.totalState["fixLoad"]+=self.interruptibleLoad.AvgPowerConsume
            
        elif mode == "unint":
            self.uninterruptibleLoad = self.unIntAgent.environment.uninterruptibleLoad
            self.uninterruptibleLoadActionMask = self.unIntAgent.states["action_mask"]
            self.totalState["unintRemain"]=self.unIntAgent.states["state"][5]
            self.totalState["unintSwitch"]=self.unIntAgent.states["state"][6]
            if self.unIntAgent.states["state"][6]==1:
                self.totalState["fixLoad"]+=self.uninterruptibleLoad.AvgPowerConsume


        #Order = 0,1,2,3,4,5
        self.totalState["order"] = (self.totalState["order"]+1 if self.totalState["order"]<5 else 0 )
        #update to next step
        if self.totalState["order"] == 0 and self.totalState["sampleTime"]!=95:
            self.totalState["sampleTime"]+=1
            self.totalState["fixLoad"]=self.Load[self.totalState["sampleTime"]]
            self.totalState["PV"]=self.PV[self.totalState["sampleTime"]]
            self.totalState["pricePerHour"]=self.GridPrice[self.totalState["sampleTime"]]
            self.totalState["deltaSoc"] = 0
            self.totalState["outdoorTemperature"]=self.outdoorTemperature[self.totalState["sampleTime"]]
            self.totalState["userSetTemperature1"]=self.userSetTemperature1[self.totalState["sampleTime"]]
            self.totalState["userSetTemperature2"]=self.userSetTemperature2[self.totalState["sampleTime"]]
            self.totalState["userSetTemperature3"]=self.userSetTemperature3[self.totalState["sampleTime"]]
            self.totalState["unintRemain"]=self.unIntAgent.environment.uninterruptibleLoad.getRemainDemand()
            self.totalState["unintSwitch"]=self.unIntAgent.environment.uninterruptibleLoad.switch
            self.totalState["intSwitch"] = 0
            self.totalState["intPreference"] = self.intUserPreference[self.totalState["sampleTime"]]
            self.totalState["unintPreference"] = self.unintPreference[self.totalState["sampleTime"]]
        
            

    def stateAbstraction(self,totalState) -> np.array:
        return np.array([totalState['sampleTime'],totalState['SOC'],totalState['fixLoad']-totalState['PV']+totalState['deltaSoc']*self.batteryCapacity,totalState['pricePerHour'],1 if totalState['userSetTemperature1']>totalState['indoorTemperature1'] or totalState['outdoorTemperature']<totalState['userSetTemperature1'] else -1,1 if totalState['userSetTemperature2']>totalState['indoorTemperature2'] or totalState['outdoorTemperature']<totalState['userSetTemperature2'] else -1,1 if totalState['userSetTemperature3']>totalState['indoorTemperature3'] or totalState['outdoorTemperature']<totalState['userSetTemperature3'] else -1,totalState['intSwitch'],totalState['unintSwitch'],totalState['intPreference'],totalState['unintPreference'],totalState['order']],dtype=np.float32)
        