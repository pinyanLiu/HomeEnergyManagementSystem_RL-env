from  lib.import_data.import_data import ImportData 
from  yaml import load , SafeLoader
from tensorforce import Environment,Agent
from lib.enviroment.multiAgentEnv.LLA.LLA import socLLA,hvacLLA,intLLA,unintLLA 
from lib.enviroment.multiAgentEnv import voidHvacTestEnv,voidInterruptibleLoadTestEnv,voidSocTestEnv,voidUnInterruptibleLoadTestEnv
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
        self.allStatisticalData = self.info.importStatisticalData()

        #construct all LLAs
        self.socAgent = socLLA(environment=voidSocTestEnv,agent=Agent.load(directory = 'Soc/saver_dir',environment=voidSocTestEnv,),mean=self.allStatisticalData.loc["SOC","mean"],std=self.allStatisticalData.loc["SOC","std"])
        
        self.hvacAgent = hvacLLA(environment=voidHvacTestEnv,agent=Agent.load(directory = 'HVAC/saver_dir',environment=voidHvacTestEnv,),mean=self.allStatisticalData.loc["HVAC","mean"],std=self.allStatisticalData.loc["HVAC","std"])
        
        self.intAgent = intLLA(environment=voidInterruptibleLoadTestEnv,agent=Agent.load(directory = 'Load/Interruptible/saver_dir',environment=voidInterruptibleLoadTestEnv,),mean=self.allStatisticalData.loc["Interruptible","mean"],std=self.allStatisticalData.loc["Interruptible","std"])
        
        self.unIntAgent = unintLLA(environment=voidUnInterruptibleLoadTestEnv,agent=Agent.load(directory = 'Load/UnInterruptible/saver_dir',environment=voidUnInterruptibleLoadTestEnv,),mean=self.allStatisticalData.loc["Uninterruptible","mean"],std=self.allStatisticalData.loc["Uninterruptible","std"])
        #AC/WM object
        self.interruptibleLoad = AC(demand=randint(1,49),AvgPowerConsume=1.5)
        self.uninterruptibleLoad = WM(demand=24,executePeriod=3,AvgPowerConsume=0.7)

        self.state = None
        self.totalState = {
            "sampleTime":0,
            "fixLoad":0,
            "PV":0,
            "SOC":0,
            "pricePerHour":0,
            "deltaSoc":0,
            "indoorTemperature":0,
            "outdoorTemperature":0,
            "userSetTemperature":0,
            "intRemain":0,
            "unintRemain":0,
            "unintSwitch":0,
            "order":0
        }
        self.reward = 0
        self.done = False

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
                10.0,
                #price per hour
                6.2,
                #HVAC state
                1,
                #Int state
                1,
                #Unint state
                1,
                #Order
                3,
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
                0.0,
                #pricePerHour
                0.0,
                #HVAC state
                0,
                #Int state
                0,
                #Unint state
                0,
                #Order
                0
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(lowerLimit,upperLimit,dtype=np.float32)
        return dict(type='float',shape=self.observation_space.shape,min_value=lowerLimit,max_value=upperLimit)

    def actions(self):
        return dict(type='int',num_values=4)

    def close(self):
        return super().close()

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
        reward = []
        sampleTime,soc,remain,pricePreHour,hvacState,intState,unIntState,order = self.state

        if order == 0 :
            self.action_mask = np.asarray([True,True,True,True])

        #choose one load executing in this sampleTime order .Load which has been execute in the same sampleTime would be auto block by action mask
        
        #soc
        if actions == 0 :
            self.socAgent.getState()
            self.socAgent.execute()
            self.updateTotalState("soc")
            self.socAgent.rewardStandardization()
            reward.append(self.socAgent.reward)
            self.action_mask = np.asarray(self.action_mask and [False,True,True,True])

        elif actions == 1:
            self.hvacAgent.getState()
            self.hvacAgent.execute()
            self.updateTotalState("hvac")
            self.hvacAgent.rewardStandardization()            
            reward.append(self.hvacAgent.reward)
            self.action_mask = np.asarray(self.action_mask and [True,False,True,True])

        elif actions == 2:
            self.intAgent.getState()
            self.intAgent.execute()
            self.updateTotalState("int")
            self.intAgent.rewardStandardization()
            reward.append(self.intAgent.reward)
            self.action_mask = np.asarray(self.action_mask and [True,True,False,True])
        elif actions == 3:
            self.unIntAgent.getState()
            self.unIntAgent.execute()
            self.updateTotalState("unint")
            self.unIntAgent.rewardStandardization()
            reward.append(self.unIntAgent.reward)
            self.action_mask = np.asarray(self.action_mask and [True,True,True,False])


        reward = sum(reward)
        self.state = self.stateAbstraction(self.totalState)

        #check if all day is done
        done =  bool(sampleTime == 95 and order == 3)

        states = dict(state=self.state,action_mask = self.action_mask)

        return states, done ,reward

    def stateAbstraction(self,totalState) -> np.array:
        res = []
        #sampleTime
        res.append(totalState[0])
        #SOC
        res.append(totalState[3])
        #remain power
        res.append(totalState[1]+totalState[2]+totalState[5]*self.batteryCapacity)
        #pricePerHour
        res.append(totalState[4])
        #HVAC state
        res.append(True if totalState[8]< totalState[6] else False)
        #interruptible load state
        res.append(True if totalState[9]==0 else False)
        #uninterruptible load state
        res.append(True if totalState[10]==0 else False)
        return np.array(res)

    def updateTotalState(self,mode) :
        if mode == "soc":
            self.totalState["deltaSoc"] += self.socAgent.actions*self.batteryCapacity
            self.totalState["SOC"]+=self.socAgent.actions
        elif mode == "hvac":
            self.totalState["fixLoad"]+=self.hvacAgent.actions
            self.totalState["indoorTemperature"] = self.hvacAgent.state[5]
        elif mode == "int":
            if self.intAgent.actions==1:
                self.totalState["fixLoad"]+=self.interruptibleLoad.AvgPowerConsume
        elif mode == "unint":
            if self.unIntAgent.state[6]==1:
                self.totalState["fixLoad"]+=self.uninterruptibleLoad.AvgPowerConsume
        #Order
        Order = (self.totalState["order"]+1 if self.totalState["order"]<=4 else 0 )
        #SampleTime
        sampleTime = (int(self.totalState["sampleTime"]+1) if self.totalState["order"]==0 else int(self.totalState["sampleTime"]))