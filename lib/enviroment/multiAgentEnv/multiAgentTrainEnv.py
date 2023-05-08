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
        self.intload_demand1 =  int(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='intload_demand1']['value'])[0])
        self.intload_power1 =  float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='intload_power1']['value'])[0])

        self.intload_demand2 =  int(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='intload_demand2']['value'])[0])
        self.intload_power2 =  float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='intload_power2']['value'])[0])

        self.intload_demand3 =  int(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='intload_demand3']['value'])[0])
        self.intload_power3 =  float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='intload_power3']['value'])[0])

        self.unload_demand1 =  int(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='unload_demand1']['value'])[0])
        self.unload_period1 =  int(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='unload_period1']['value'])[0])
        self.unload_power1 =  float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='unload_power1']['value'])[0])
        
        self.unload_demand2 =  int(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='unload_demand2']['value'])[0])
        self.unload_period2 =  int(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='unload_period2']['value'])[0])
        self.unload_power2 =  float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='unload_power2']['value'])[0])

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
        self.interruptibleLoad1 = AC(demand=self.intload_demand1,AvgPowerConsume=self.intload_power1)
        self.interruptibleLoad2 = AC(demand=self.intload_demand2,AvgPowerConsume=self.intload_power2)
        self.interruptibleLoad3 = AC(demand=self.intload_demand3,AvgPowerConsume=self.intload_power3)
        self.uninterruptibleLoad = WM(demand=self.unload_demand1,executePeriod=self.unload_period1,AvgPowerConsume=self.unload_power1)
        self.uninterruptibleLoad = WM(demand=self.unload_demand2,executePeriod=self.unload_period2,AvgPowerConsume=self.unload_power2)

        # self.interruptibleLoad = AC(demand=randint(1,49),AvgPowerConsume=1.5)
        # self.uninterruptibleLoad = WM(demand=randint(3,24),executePeriod=3,AvgPowerConsume=uniform(0.5,1))
        self.allIntPreference1 = self.info.importIntPreference(1)
        self.allIntPreference2 = self.info.importIntPreference(2)
        self.allIntPreference3 = self.info.importIntPreference(3)
        self.allUnintPreference1 = self.info.importUnIntPreference(1)
        self.allUnintPreference2 = self.info.importUnIntPreference(2)



        #construct all LLAs
        
        self.socAgent = socLLA(mean=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='SOC']['mean'])[0]),std=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='SOC']['std'])[0]),min=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='SOC']['Min'])[0]),max=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='SOC']['Max'])[0]),baseParameter=self.BaseParameter)

        
        self.hvacAgent1 = hvacLLA(mean=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='HVAC']['mean'])[0]),std=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='HVAC']['std'])[0]),min=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='HVAC']['Min'])[0]),max=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='HVAC']['Max'])[0]),baseParameter=self.BaseParameter,allOutdoorTemperature=self.allOutdoorTemperature,allUserSetTemperature=self.allUserSetTemperature1,id=1)

        self.hvacAgent2 = hvacLLA(mean=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='HVAC']['mean'])[0]),std=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='HVAC']['std'])[0]),min=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='HVAC']['Min'])[0]),max=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='HVAC']['Max'])[0]),baseParameter=self.BaseParameter,allOutdoorTemperature=self.allOutdoorTemperature,allUserSetTemperature=self.allUserSetTemperature2,id=2)

        self.hvacAgent3 = hvacLLA(mean=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='HVAC']['mean'])[0]),std=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='HVAC']['std'])[0]),min=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='HVAC']['Min'])[0]),max=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='HVAC']['Max'])[0]),baseParameter=self.BaseParameter,allOutdoorTemperature=self.allOutdoorTemperature,allUserSetTemperature=self.allUserSetTemperature3,id=3)
        
        
        self.intAgent1 = intLLA(mean=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='Interruptible']['mean'])[0]),std=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='Interruptible']['std'])[0]),min=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='Interruptible']['Min'])[0]),max=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='Interruptible']['Max'])[0]),baseParameter=self.BaseParameter,Int=self.interruptibleLoad1)

        self.intAgent2 = intLLA(mean=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='Interruptible']['mean'])[0]),std=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='Interruptible']['std'])[0]),min=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='Interruptible']['Min'])[0]),max=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='Interruptible']['Max'])[0]),baseParameter=self.BaseParameter,Int=self.interruptibleLoad2)

        self.intAgent3 = intLLA(mean=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='Interruptible']['mean'])[0]),std=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='Interruptible']['std'])[0]),min=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='Interruptible']['Min'])[0]),max=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='Interruptible']['Max'])[0]),baseParameter=self.BaseParameter,Int=self.interruptibleLoad3)
        
        
        self.unIntAgent1 = unintLLA(mean=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='Uninterruptible']['mean'])[0]),std=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='Uninterruptible']['std'])[0]),min=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='Uninterruptible']['Min'])[0]),max=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='Uninterruptible']['Max'])[0]),baseParameter=self.BaseParameter,unInt=self.uninterruptibleLoad1)

        self.unIntAgent2 = unintLLA(mean=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='Uninterruptible']['mean'])[0]),std=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='Uninterruptible']['std'])[0]),min=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='Uninterruptible']['Min'])[0]),max=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='Uninterruptible']['Max'])[0]),baseParameter=self.BaseParameter,unInt=self.uninterruptibleLoad2)


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
            "intSwitch1":0,
            "intRemain1":0,
            "intPreference1":0,
            "intSwitch2":0,
            "intRemain2":0,
            "intPreference2":0,
            "intSwitch3":0,
            "intRemain3":0,
            "intPreference3":0,
            "unintRemain1":0,
            "unintSwitch1":0,
            "unintPreference1":0,
            "unintRemain2":0,
            "unintSwitch2":0,
            "unintPreference2":0,
            "order":0,
            "PgridMax":0
        }
        self.reward = 0
        self.done = False
        self.action_mask = [True,True,True,True,True,True,True,True,True,True]
        self.interruptibleLoadActionMask1 = [True,True]
        self.interruptibleLoadActionMask2 = [True,True]
        self.interruptibleLoadActionMask3 = [True,True]
        self.uninterruptibleLoadActionMask1 = [True,True]
        self.uninterruptibleLoadActionMask2 = [True,True]
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
                #Int1 state
                1,
                #Int2 state
                1,
                #Int3 state
                1,
                #Unint1 state
                1,
                #Unint2 state
                1,
                #Int1 preference
                4,
                #Int2 preference
                4,
                #Int3 preference
                4,
                #unInt1 preference
                4,
                #unInt2 preference
                4,
                #Order
                9
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
                #Int1 state
                0,
                #Int2 state
                0,
                #Int3 state
                0,
                #Unint1 state
                0,
                #Unint2 state
                0,
                #Int1 preference
                -1,
                #Int2 preference
                -1,
                #Int3 preference
                -1,
                #unInt1 preference
                -1,
                #unInt2 preference
                -1,
                #Order
                0
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(lowerLimit,upperLimit,dtype=np.float32)
        return dict(type='float',shape=self.observation_space.shape,min_value=lowerLimit,max_value=upperLimit)

    def actions(self):
        return dict(type='int',num_values=10)

    def close(self):
        return super().close()

    def reset(self):
        '''
        Starting State
        '''
        #pick one day from 360 days
        self.i = randint(1,359)
        #AC/WM object

        self.interruptibleLoad1 = AC(demand=randint(15,25),AvgPowerConsume=uniform(1,1.5))

        self.interruptibleLoad2 = AC(demand=randint(15,25),AvgPowerConsume=uniform(1,1.5))

        self.interruptibleLoad3 = AC(demand=randint(15,25),AvgPowerConsume=uniform(1,1.5))

        self.uninterruptibleLoad1 = WM(demand=randint(3,8),executePeriod=6,AvgPowerConsume=uniform(0.8,1.5))

        self.uninterruptibleLoad2 = WM(demand=randint(3,8),executePeriod=3,AvgPowerConsume=uniform(0.8,1.5))
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
            self.intUserPreference1 = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference1['1'].tolist(),self.randomDeltaPreference)]
            self.intUserPreference2 = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference2['1'].tolist(),self.randomDeltaPreference)]
            self.intUserPreference3 = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference3['1'].tolist(),self.randomDeltaPreference)]
            self.unintPreference1 = [min(max(x+y,-1),4) for x,y in zip(self.allUnintPreference1['1'].tolist(),self.randomDeltaPreference)]
            self.unintPreference2 = [min(max(x+y,-1),4) for x,y in zip(self.allUnintPreference2['1'].tolist(),self.randomDeltaPreference)]

            
        elif int(self.i / 30) == 1:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['Feb'].tolist(),self.randomTemperature)]
            self.userSetTemperature1 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature1['Feb'].tolist(),self.randomTemperature)]
            self.userSetTemperature2 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature2['Feb'].tolist(),self.randomTemperature)]
            self.userSetTemperature3 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature3['Feb'].tolist(),self.randomTemperature)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Feb'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
            self.intUserPreference1 = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference1['2'].tolist(),self.randomDeltaPreference)]
            self.intUserPreference2 = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference2['2'].tolist(),self.randomDeltaPreference)]
            self.intUserPreference3 = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference3['2'].tolist(),self.randomDeltaPreference)]
            self.unintPreference1 = [min(max(x+y,-1),4) for x,y in zip(self.allUnintPreference1['2'].tolist(),self.randomDeltaPreference)]
            self.unintPreference2 = [min(max(x+y,-1),4) for x,y in zip(self.allUnintPreference2['2'].tolist(),self.randomDeltaPreference)]

        elif int(self.i / 30) == 2:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['Mar'].tolist(),self.randomTemperature)]
            self.userSetTemperature1 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature1['Mar'].tolist(),self.randomTemperature)]
            self.userSetTemperature2 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature2['Mar'].tolist(),self.randomTemperature)]
            self.userSetTemperature3 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature3['Mar'].tolist(),self.randomTemperature)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Mar'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
            self.intUserPreference1 = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference1['3'].tolist(),self.randomDeltaPreference)]
            self.intUserPreference2 = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference2['3'].tolist(),self.randomDeltaPreference)]
            self.intUserPreference3 = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference3['3'].tolist(),self.randomDeltaPreference)]
            self.unintPreference1 = [min(max(x+y,-1),4) for x,y in zip(self.allUnintPreference1['3'].tolist(),self.randomDeltaPreference)]
            self.unintPreference2 = [min(max(x+y,-1),4) for x,y in zip(self.allUnintPreference2['3'].tolist(),self.randomDeltaPreference)]

        elif int(self.i / 30) == 3:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['Apr'].tolist(),self.randomTemperature)]
            self.userSetTemperature1 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature1['Apr'].tolist(),self.randomTemperature)]
            self.userSetTemperature2 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature2['Apr'].tolist(),self.randomTemperature)]
            self.userSetTemperature3 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature3['Apr'].tolist(),self.randomTemperature)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Apr'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
            self.intUserPreference1 = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference1['4'].tolist(),self.randomDeltaPreference)]
            self.intUserPreference2 = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference2['4'].tolist(),self.randomDeltaPreference)]
            self.intUserPreference3 = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference3['4'].tolist(),self.randomDeltaPreference)]
            self.unintPreference1 = [min(max(x+y,-1),4) for x,y in zip(self.allUnintPreference1['4'].tolist(),self.randomDeltaPreference)]
            self.unintPreference2 = [min(max(x+y,-1),4) for x,y in zip(self.allUnintPreference2['4'].tolist(),self.randomDeltaPreference)]


        elif int(self.i / 30) == 4:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['May'].tolist(),self.randomTemperature)]
            self.userSetTemperature1 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature1['May'].tolist(),self.randomTemperature)]
            self.userSetTemperature2 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature2['May'].tolist(),self.randomTemperature)]
            self.userSetTemperature3 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature3['May'].tolist(),self.randomTemperature)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['May'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
            self.intUserPreference1 = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference1['5'].tolist(),self.randomDeltaPreference)]
            self.intUserPreference2 = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference2['5'].tolist(),self.randomDeltaPreference)]
            self.intUserPreference3 = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference3['5'].tolist(),self.randomDeltaPreference)]
            self.unintPreference1 = [min(max(x+y,-1),4) for x,y in zip(self.allUnintPreference1['5'].tolist(),self.randomDeltaPreference)]
            self.unintPreference2 = [min(max(x+y,-1),4) for x,y in zip(self.allUnintPreference2['5'].tolist(),self.randomDeltaPreference)]

        elif int(self.i / 30) == 5:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['Jun'].tolist(),self.randomTemperature)]
            self.userSetTemperature1 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature1['Jun'].tolist(),self.randomTemperature)]
            self.userSetTemperature2 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature2['Jun'].tolist(),self.randomTemperature)]
            self.userSetTemperature3 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature3['Jun'].tolist(),self.randomTemperature)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Jun'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
            self.intUserPreference1 = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference1['6'].tolist(),self.randomDeltaPreference)]
            self.intUserPreference2 = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference2['6'].tolist(),self.randomDeltaPreference)]
            self.intUserPreference3 = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference3['6'].tolist(),self.randomDeltaPreference)]
            self.unintPreference1 = [min(max(x+y,-1),4) for x,y in zip(self.allUnintPreference1['6'].tolist(),self.randomDeltaPreference)]
            self.unintPreference2 = [min(max(x+y,-1),4) for x,y in zip(self.allUnintPreference2['6'].tolist(),self.randomDeltaPreference)]

        elif int(self.i / 30) == 6:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['July'].tolist(),self.randomTemperature)]
            self.userSetTemperature1 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature1['July'].tolist(),self.randomTemperature)]
            self.userSetTemperature2 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature2['July'].tolist(),self.randomTemperature)]
            self.userSetTemperature3 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature3['July'].tolist(),self.randomTemperature)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['July'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.summerGridPrice,self.randomDeltaPrice) ]
            self.intUserPreference1 = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference1['7'].tolist(),self.randomDeltaPreference)]
            self.intUserPreference2 = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference2['7'].tolist(),self.randomDeltaPreference)]
            self.intUserPreference3 = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference3['7'].tolist(),self.randomDeltaPreference)]
            self.unintPreference1 = [min(max(x+y,-1),4) for x,y in zip(self.allUnintPreference1['7'].tolist(),self.randomDeltaPreference)]
            self.unintPreference2 = [min(max(x+y,-1),4) for x,y in zip(self.allUnintPreference2['7'].tolist(),self.randomDeltaPreference)]

        elif int(self.i / 30) == 7:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['Aug'].tolist(),self.randomTemperature)]
            self.userSetTemperature1 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature1['Aug'].tolist(),self.randomTemperature)]
            self.userSetTemperature2 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature2['Aug'].tolist(),self.randomTemperature)]
            self.userSetTemperature3 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature3['Aug'].tolist(),self.randomTemperature)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Aug'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.summerGridPrice,self.randomDeltaPrice) ]
            self.intUserPreference1 = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference1['8'].tolist(),self.randomDeltaPreference)]
            self.intUserPreference2 = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference2['8'].tolist(),self.randomDeltaPreference)]
            self.intUserPreference3 = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference3['8'].tolist(),self.randomDeltaPreference)]
            self.unintPreference1 = [min(max(x+y,-1),4) for x,y in zip(self.allUnintPreference1['8'].tolist(),self.randomDeltaPreference)]
            self.unintPreference2 = [min(max(x+y,-1),4) for x,y in zip(self.allUnintPreference2['8'].tolist(),self.randomDeltaPreference)]

        elif int(self.i / 30) == 8:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['Sep'].tolist(),self.randomTemperature)]
            self.userSetTemperature1 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature1['Sep'].tolist(),self.randomTemperature)]
            self.userSetTemperature2 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature2['Sep'].tolist(),self.randomTemperature)]
            self.userSetTemperature3 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature3['Sep'].tolist(),self.randomTemperature)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Sep'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.summerGridPrice,self.randomDeltaPrice) ]
            self.intUserPreference1 = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference1['9'].tolist(),self.randomDeltaPreference)]
            self.intUserPreference2 = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference2['9'].tolist(),self.randomDeltaPreference)]
            self.intUserPreference3 = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference3['9'].tolist(),self.randomDeltaPreference)]
            self.unintPreference1 = [min(max(x+y,-1),4) for x,y in zip(self.allUnintPreference1['9'].tolist(),self.randomDeltaPreference)]
            self.unintPreference2 = [min(max(x+y,-1),4) for x,y in zip(self.allUnintPreference2['9'].tolist(),self.randomDeltaPreference)]

        elif int(self.i / 30) == 9:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['Oct'].tolist(),self.randomTemperature)]
            self.userSetTemperature1 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature1['Oct'].tolist(),self.randomTemperature)]
            self.userSetTemperature2 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature2['Oct'].tolist(),self.randomTemperature)]
            self.userSetTemperature3 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature3['Oct'].tolist(),self.randomTemperature)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Oct'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.summerGridPrice,self.randomDeltaPrice) ]
            self.intUserPreference1 = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference1['10'].tolist(),self.randomDeltaPreference)]
            self.intUserPreference2 = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference2['10'].tolist(),self.randomDeltaPreference)]
            self.intUserPreference3 = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference3['10'].tolist(),self.randomDeltaPreference)]
            self.unintPreference1 = [min(max(x+y,-1),4) for x,y in zip(self.allUnintPreference1['10'].tolist(),self.randomDeltaPreference)]
            self.unintPreference2 = [min(max(x+y,-1),4) for x,y in zip(self.allUnintPreference2['10'].tolist(),self.randomDeltaPreference)]

        elif int(self.i / 30) == 10:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['Nov'].tolist(),self.randomTemperature)]
            self.userSetTemperature1 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature1['Nov'].tolist(),self.randomTemperature)]
            self.userSetTemperature2 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature2['Nov'].tolist(),self.randomTemperature)]
            self.userSetTemperature3 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature3['Nov'].tolist(),self.randomTemperature)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Nov'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
            self.intUserPreference1 = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference1['11'].tolist(),self.randomDeltaPreference)]
            self.intUserPreference2 = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference2['11'].tolist(),self.randomDeltaPreference)]
            self.intUserPreference3 = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference3['11'].tolist(),self.randomDeltaPreference)]
            self.unintPreference1 = [min(max(x+y,-1),4) for x,y in zip(self.allUnintPreference1['11'].tolist(),self.randomDeltaPreference)]
            self.unintPreference2 = [min(max(x+y,-1),4) for x,y in zip(self.allUnintPreference2['11'].tolist(),self.randomDeltaPreference)]

        elif int(self.i / 30) == 11:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['Dcb'].tolist(),self.randomTemperature)]
            self.userSetTemperature1 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature1['Dcb'].tolist(),self.randomTemperature)]
            self.userSetTemperature2 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature2['Dcb'].tolist(),self.randomTemperature)]
            self.userSetTemperature3 = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature3['Dcb'].tolist(),self.randomTemperature)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Dec'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
            self.intUserPreference1 = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference1['12'].tolist(),self.randomDeltaPreference)]
            self.intUserPreference2 = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference2['12'].tolist(),self.randomDeltaPreference)]
            self.intUserPreference3 = [min(max(x+y,-1),4) for x,y in zip(self.allIntPreference3['12'].tolist(),self.randomDeltaPreference)]
            self.unintPreference1 = [min(max(x+y,-1),4) for x,y in zip(self.allUnintPreference1['12'].tolist(),self.randomDeltaPreference)]
            self.unintPreference2 = [min(max(x+y,-1),4) for x,y in zip(self.allUnintPreference2['12'].tolist(),self.randomDeltaPreference)]



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
            "intRemain":self.interruptibleLoad1.demand,
            "intSwitch":self.interruptibleLoad1.switch,
            "intPreference":self.intUserPreference1[0],
            "intRemain":self.interruptibleLoad2.demand,
            "intSwitch":self.interruptibleLoad2.switch,
            "intPreference":self.intUserPreference2[0],
            "intRemain":self.interruptibleLoad3.demand,
            "intSwitch":self.interruptibleLoad3.switch,
            "intPreference":self.intUserPreference3[0],
            "unintRemain":self.uninterruptibleLoad1.demand*self.uninterruptibleLoad1.executePeriod,
            "unintSwitch":self.uninterruptibleLoad1.switch,
            "unintPreference":self.unintPreference1[0],
            "unintRemain":self.uninterruptibleLoad2.demand*self.uninterruptibleLoad2.executePeriod,
            "unintSwitch":self.uninterruptibleLoad2.switch,
            "unintPreference":self.unintPreference2[0],
            "order":0,
            "PgridMax":self.PgridMax
        }
        self.interruptibleLoadActionMask = [True,True]
        self.uninterruptibleLoadActionMask = [True,True]
        self.action_mask = [True,True,True,True,True,True,True,True,True,True]
        self.state = self.stateAbstraction(self.totalState)
        self.socAgent.agent.internals = self.socAgent.agent.initial_internals()
        self.hvacAgent1.agent.internals = self.hvacAgent1.agent.initial_internals()
        self.hvacAgent2.agent.internals = self.hvacAgent2.agent.initial_internals()
        self.hvacAgent3.agent.internals = self.hvacAgent3.agent.initial_internals()
        self.intAgent1.agent.internals = self.intAgent1.agent.initial_internals()
        self.intAgent2.agent.internals = self.intAgent2.agent.initial_internals()
        self.intAgent3.agent.internals = self.intAgent3.agent.initial_internals()
        self.unIntAgent1.agent.internals = self.unIntAgent1.agent.initial_internals()
        self.unIntAgent2.agent.internals = self.unIntAgent2.agent.initial_internals()
        self.socAgent.environment.reset()
        self.hvacAgent1.environment.reset()
        self.hvacAgent2.environment.reset()
        self.hvacAgent3.environment.reset()
        self.intAgent1.environment.reset()
        self.intAgent2.environment.reset()
        self.intAgent3.environment.reset()
        self.unIntAgent1.environment.reset()
        self.unIntAgent2.environment.reset()


        return self.state


    def execute(self,actions):
        '''
        interaction of each state(changes while taking action)
        Rewards
        Episode Termination condition
        '''
        reward = []
        sampleTime,soc,remain,pricePreHour,hvacState1,hvacState2,hvacState3,intState1,intState2,intState3,unIntState1,unIntState2,intPreference1,intPreference2,intPreference3,unintPreference1,unintPreference2,order = self.state
#         #print(self.totalState)

        #choose one load executing in this sampleTime order .Load which has been execute in the same sampleTime would be auto block by action mask

        #soc
        if actions == 0 :
            # print('soc')
            self.socAgent.getState(self.totalState)
            self.socAgent.environment.updateState(self.socAgent.states)
            self.socAgent.execute()
            self.socAgent.rewardNormalization()
            reward.append(self.socAgent.reward)
            self.updateTotalState("soc")
            self.action_mask = [a and b for a,b in zip(self.action_mask , [False,True,True,True,True,True,True,True,True,True])]
        #hvac1
        elif actions == 1:
            # print('hvac')
            self.hvacAgent1.getState(self.totalState)
            self.hvacAgent1.environment.updateState(self.hvacAgent1.states)
            self.hvacAgent1.execute()
            self.hvacAgent1.rewardNormalization()
            reward.append(self.hvacAgent1.reward)            
            self.updateTotalState("hvac1")
            self.action_mask = [a and b for a,b in zip(self.action_mask , [True,False,True,True,True,True,True,True,True,True])]
        #hvac2
        elif actions == 2:
            # print('hvac')
            self.hvacAgent2.getState(self.totalState)
            self.hvacAgent2.environment.updateState(self.hvacAgent2.states)
            self.hvacAgent2.execute()
            self.hvacAgent2.rewardNormalization()
            reward.append(self.hvacAgent2.reward)            
            self.updateTotalState("hvac2")
            self.action_mask = [a and b for a,b in zip(self.action_mask , [True,True,False,True,True,True,True,True,True,True])]
        #hvac3
        elif actions == 3:
            # print('hvac')
            self.hvacAgent3.getState(self.totalState)
            self.hvacAgent3.environment.updateState(self.hvacAgent3.states)
            self.hvacAgent3.execute()
            self.hvacAgent3.rewardNormalization()
            reward.append(self.hvacAgent3.reward)            
            self.updateTotalState("hvac3")
            self.action_mask = [a and b for a,b in zip(self.action_mask , [True,True,True,False,True,True,True,True,True,True])]
        
        #int
        elif actions == 4:
            # print('int')
            self.intAgent1.getState(self.totalState,self.interruptibleLoadActionMask)
            self.intAgent1.environment.updateState(self.intAgent1.states,self.interruptibleLoad)
            self.intAgent1.execute()
            self.intAgent1.rewardNormalization()
            reward.append(self.intAgent1.reward)            
            self.updateTotalState("int1")
            self.action_mask = [a and b for a,b in zip(self.action_mask , [True,True,True,True,False,True,True,True,True,True])]
        elif actions == 5:
            # print('int')
            self.intAgent2.getState(self.totalState,self.interruptibleLoadActionMask)
            self.intAgent2.environment.updateState(self.intAgent2.states,self.interruptibleLoad)
            self.intAgent2.execute()
            self.intAgent2.rewardNormalization()
            reward.append(self.intAgent2.reward)            
            self.updateTotalState("int2")
            self.action_mask = [a and b for a,b in zip(self.action_mask , [True,True,True,True,True,False,True,True,True,True])]
        elif actions == 6:
            # print('int')
            self.intAgent3.getState(self.totalState,self.interruptibleLoadActionMask)
            self.intAgent3.environment.updateState(self.intAgent3.states,self.interruptibleLoad)
            self.intAgent3.execute()
            self.intAgent3.rewardNormalization()
            reward.append(self.intAgent3.reward)            
            self.updateTotalState("int3")
            self.action_mask = [a and b for a,b in zip(self.action_mask , [True,True,True,True,True,True,False,True,True,True])]
        #unint
        elif actions == 7:
            # print('unint')
            self.unIntAgent1.getState(self.totalState,self.uninterruptibleLoadActionMask)
            self.unIntAgent1.environment.updateState(self.unIntAgent1.states,self.uninterruptibleLoad)
            self.unIntAgent1.execute()
            self.unIntAgent1.rewardNormalization()
            reward.append(self.unIntAgent1.reward)
            self.updateTotalState("unint1")
            self.action_mask = [a and b for a,b in zip(self.action_mask , [True,True,True,True,True,True,True,False,True,True])]
        #unint
        elif actions == 8:
            # print('unint')
            self.unIntAgent2.getState(self.totalState,self.uninterruptibleLoadActionMask)
            self.unIntAgent2.environment.updateState(self.unIntAgent2.states,self.uninterruptibleLoad)
            self.unIntAgent2.execute()
            self.unIntAgent2.rewardNormalization()
            reward.append(self.unIntAgent2.reward)
            self.updateTotalState("unint2")
            self.action_mask = [a and b for a,b in zip(self.action_mask , [True,True,True,True,True,True,True,True,False,True])]
        #none
        else:
            # print('none')
            self.updateTotalState("None")
            reward.append(-1)
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
            self.action_mask = [a and b for a,b in zip(self.action_mask , [True,False,False,False,False,False,False,False,False,True])]
        

        if order == 8:
            if self.action_mask[1] == True:
                self.totalState["indoorTemperature1"] = self.epsilon*self.totalState["indoorTemperature1"]+(1-self.epsilon)*(self.totalState["outdoorTemperature"])
            if self.action_mask[2] == True:
                self.totalState["indoorTemperature2"] = self.epsilon*self.totalState["indoorTemperature2"]+(1-self.epsilon)*(self.totalState["outdoorTemperature"])
            if self.action_mask[3] == True:
                self.totalState["indoorTemperature3"] = self.epsilon*self.totalState["indoorTemperature3"]+(1-self.epsilon)*(self.totalState["outdoorTemperature"])
            if self.action_mask[7]==True:
                self.unIntAgent1.environment.uninterruptibleLoad.step()
            if self.action_mask[8]==True:
                self.unIntAgent2.environment.uninterruptibleLoad.step()
            self.action_mask = [False,False,False,False,False,False,False,False,False,True]
            if(self.state[2]-0.25>self.PgridMax):  
                # print("PGRID MAX OVER!!!")
                reward.append(-10)
            else:
                reward.append(0.0001)

        if order==9:
            self.action_mask = [True,True,True,True,True,True,True,True,True,True]



        #check if all day is done
        done =  bool(sampleTime == 95 and order == 9)

        reward = sum(reward)
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

        elif mode == "int1":
            self.interruptibleLoad = self.intAgent1.environment.interruptibleLoad
            self.interruptibleLoadActionMask = self.intAgent1.states["action_mask"]
            self.totalState["intRemain"] = self.intAgent1.states["state"][5]
            self.totalState["intSwitch"] = self.intAgent1.actions
            if self.intAgent1.actions==1:
                self.totalState["fixLoad"]+=self.interruptibleLoad.AvgPowerConsume

        elif mode == "int2":
            self.interruptibleLoad = self.intAgent2.environment.interruptibleLoad
            self.interruptibleLoadActionMask = self.intAgent2.states["action_mask"]
            self.totalState["intRemain"] = self.intAgent2.states["state"][5]
            self.totalState["intSwitch"] = self.intAgent2.actions
            if self.intAgent2.actions==1:
                self.totalState["fixLoad"]+=self.interruptibleLoad.AvgPowerConsume

        elif mode == "int3":
            self.interruptibleLoad = self.intAgent3.environment.interruptibleLoad
            self.interruptibleLoadActionMask = self.intAgent3.states["action_mask"]
            self.totalState["intRemain"] = self.intAgent3.states["state"][5]
            self.totalState["intSwitch"] = self.intAgent3.actions
            if self.intAgent3.actions==1:
                self.totalState["fixLoad"]+=self.interruptibleLoad.AvgPowerConsume
            
        elif mode == "unint1":
            self.uninterruptibleLoad = self.unIntAgent1.environment.uninterruptibleLoad
            self.uninterruptibleLoadActionMask = self.unIntAgent1.states["action_mask"]
            self.totalState["unintRemain"]=self.unIntAgent1.states["state"][5]
            self.totalState["unintSwitch"]=self.unIntAgent1.states["state"][6]
            if self.unIntAgent1.states["state"][6]==1:
                self.totalState["fixLoad"]+=self.uninterruptibleLoad.AvgPowerConsume
        
        elif mode == "unint2":
            self.uninterruptibleLoad = self.unIntAgent2.environment.uninterruptibleLoad
            self.uninterruptibleLoadActionMask = self.unIntAgent2.states["action_mask"]
            self.totalState["unintRemain"]=self.unIntAgent2.states["state"][5]
            self.totalState["unintSwitch"]=self.unIntAgent2.states["state"][6]
            if self.unIntAgent2.states["state"][6]==1:
                self.totalState["fixLoad"]+=self.uninterruptibleLoad.AvgPowerConsume


        #Order = 0,1,2,3,4,5,6 
        self.totalState["order"] = (self.totalState["order"]+1 if self.totalState["order"]<=8 else 0 )
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
            self.totalState["unintRemain1"]=self.unIntAgent1.environment.uninterruptibleLoad.getRemainDemand()
            self.totalState["unintRemain2"]=self.unIntAgent2.environment.uninterruptibleLoad.getRemainDemand()
            self.totalState["unintSwitch1"]=self.unIntAgent1.environment.uninterruptibleLoad.switch
            self.totalState["unintSwitch2"]=self.unIntAgent2.environment.uninterruptibleLoad.switch
            self.totalState["intSwitch1"] = 0
            self.totalState["intSwitch2"] = 0
            self.totalState["intSwitch3"] = 0
            self.totalState["intPreference1"] = self.intUserPreference1[self.totalState["sampleTime"]]
            self.totalState["intPreference2"] = self.intUserPreference2[self.totalState["sampleTime"]]
            self.totalState["intPreference3"] = self.intUserPreference3[self.totalState["sampleTime"]]
            self.totalState["unintPreference1"] = self.unintPreference1[self.totalState["sampleTime"]]
            self.totalState["unintPreference2"] = self.unintPreference2[self.totalState["sampleTime"]]
        
            

    def stateAbstraction(self,totalState) -> np.array:
        return np.array([totalState['sampleTime'],totalState['SOC'],totalState['fixLoad']-totalState['PV']+totalState['deltaSoc']*self.batteryCapacity,totalState['pricePerHour'],1 if totalState['userSetTemperature1']>totalState['indoorTemperature1'] or totalState['outdoorTemperature']<totalState['userSetTemperature1'] else -1,1 if totalState['userSetTemperature2']>totalState['indoorTemperature2'] or totalState['outdoorTemperature']<totalState['userSetTemperature2'] else -1,1 if totalState['userSetTemperature3']>totalState['indoorTemperature3'] or totalState['outdoorTemperature']<totalState['userSetTemperature3'] else -1,totalState['intSwitch1'],totalState['intSwitch2'],totalState['intSwitch3'],totalState['unintSwitch1'],totalState['unintSwitch2'],totalState['intPreference1'],totalState['intPreference2'],totalState['intPreference3'],totalState['unintPreference1'],totalState['unintPreference2'],totalState['order']],dtype=np.float32)
        