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
        #self.notSummerGridPrice = self.allGridPrice['not_summer_price'].tolist()
        self.notSummerGridPrice = self.allGridPrice['test_price1'].tolist()
        #pick one day from training data
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

        #unint  / int object 
        self.interruptibleLoad = AC(demand=randint(1,49),AvgPowerConsume=1.5)
        self.uninterruptibleLoad = WM(demand=randint(3,24),executePeriod=3,AvgPowerConsume=uniform(0.5,1))


        #construct all LLAs
        
        self.socAgent = socLLA(mean=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='SOC']['mean'])[0]),std=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='SOC']['std'])[0]),baseParameter=self.BaseParameter)
        self.socAgent.environment.reset()

        
        self.hvacAgent = hvacLLA(mean=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='HVAC']['mean'])[0]),std=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='HVAC']['std'])[0]))
        
        
        self.intAgent = intLLA(mean=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='Interruptible']['mean'])[0]),std=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='Interruptible']['std'])[0]),Int=self.interruptibleLoad)
        
        
        self.unIntAgent = unintLLA(mean=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='Uninterruptible']['mean'])[0]),std=float(list(self.allStatisticalData.loc[self.allStatisticalData['name']=='Uninterruptible']['std'])[0]),unInt=self.uninterruptibleLoad)


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
                -10.0,
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
        #pick one day from 360 days
        self.i = randint(1,359)
        #AC/WM object
        self.interruptibleLoad = AC(demand=randint(1,49),AvgPowerConsume=1.5)
        self.uninterruptibleLoad = WM(demand=randint(3,24),executePeriod=3,AvgPowerConsume=uniform(0.5,1))
        self.randomDeltaPrice  = [uniform(-1,1) for _ in range(96)]
        self.randomDeltaPV = [uniform(-0.5,0.5) for _ in range(96)]        
        self.randomTemperature = [uniform(-2,2)for _ in range(96)]

        #import PV,outTmp,userTmp,GridPrice
        if int( self.i / 30) == 0:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['Jan'].tolist(),self.randomTemperature)]
            self.userSetTemperature = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature['Jan'].tolist(),self.randomTemperature)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Jan'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
        elif int(self.i / 30) == 1:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['Feb'].tolist(),self.randomTemperature)]
            self.userSetTemperature = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature['Feb'].tolist(),self.randomTemperature)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Feb'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
        elif int(self.i / 30) == 2:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['Mar'].tolist(),self.randomTemperature)]
            self.userSetTemperature = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature['Mar'].tolist(),self.randomTemperature)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Mar'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
        elif int(self.i / 30) == 3:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['Apr'].tolist(),self.randomTemperature)]
            self.userSetTemperature = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature['Apr'].tolist(),self.randomTemperature)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Apr'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
        elif int(self.i / 30) == 4:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['May'].tolist(),self.randomTemperature)]
            self.userSetTemperature = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature['May'].tolist(),self.randomTemperature)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['May'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
        elif int(self.i / 30) == 5:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['Jun'].tolist(),self.randomTemperature)]
            self.userSetTemperature = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature['Jun'].tolist(),self.randomTemperature)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Jun'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
        elif int(self.i / 30) == 6:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['July'].tolist(),self.randomTemperature)]
            self.userSetTemperature = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature['July'].tolist(),self.randomTemperature)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['July'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.summerGridPrice,self.randomDeltaPrice) ]
        elif int(self.i / 30) == 7:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['Aug'].tolist(),self.randomTemperature)]
            self.userSetTemperature = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature['Aug'].tolist(),self.randomTemperature)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Aug'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.summerGridPrice,self.randomDeltaPrice) ]
        elif int(self.i / 30) == 8:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['Sep'].tolist(),self.randomTemperature)]
            self.userSetTemperature = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature['Sep'].tolist(),self.randomTemperature)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Sep'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.summerGridPrice,self.randomDeltaPrice) ]
        elif int(self.i / 30) == 9:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['Oct'].tolist(),self.randomTemperature)]
            self.userSetTemperature = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature['Oct'].tolist(),self.randomTemperature)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Oct'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.summerGridPrice,self.randomDeltaPrice) ]
        elif int(self.i / 30) == 10:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['Nov'].tolist(),self.randomTemperature)]
            self.userSetTemperature = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature['Nov'].tolist(),self.randomTemperature)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Nov'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]
        elif int(self.i / 30) == 11:
            self.outdoorTemperature = [min(max(x+y,35),104) for x,y in zip(self.allOutdoorTemperature['Dcb'].tolist(),self.randomTemperature)]
            self.userSetTemperature = [min(max(x+y,35),104) for x,y in zip(self.allUserSetTemperature['Dcb'].tolist(),self.randomTemperature)]
            self.PV = [min(max(x+y,0),10) for x,y in zip(self.allPV['Dec'].tolist(),self.randomDeltaPV)]
            self.GridPrice = [min(max(x+y,0),6.2) for x,y in zip(self.notSummerGridPrice,self.randomDeltaPrice) ]

        #reset state
        self.totalState = {
            "sampleTime":0,
            "fixLoad":self.Load[0],
            "PV":self.PV[0],
            "SOC":self.socInit,
            "pricePerHour":self.GridPrice[0],
            "deltaSoc":0,
            "indoorTemperature":self.initIndoorTemperature,
            "outdoorTemperature":self.outdoorTemperature[0],
            "userSetTemperature":self.userSetTemperature[0],
            "intRemain":self.interruptibleLoad.demand,
            "unintRemain":self.uninterruptibleLoad.demand,
            "unintSwitch":self.uninterruptibleLoad.switch,
            "order":0
        }

        self.state = self.stateAbstraction(self.totalState)
        return self.state


    def execute(self,actions):
        '''
        interaction of each state(changes while taking action)
        Rewards
        Episode Termination condition
        '''
        reward = []
        sampleTime,soc,remain,pricePreHour,hvacState,intState,unIntState,order = self.state

        if order == 0 :
            self.action_mask = [True,True,True,True]

        #choose one load executing in this sampleTime order .Load which has been execute in the same sampleTime would be auto block by action mask
        
        #soc
        if actions == 0 :
            self.socAgent.getState(self.totalState)
            self.socAgent.environment.updateState(self.socAgent.states)
            self.socAgent.execute()
            self.updateTotalState("soc")
            self.socAgent.rewardStandardization()
            reward.append(self.socAgent.reward)
            self.action_mask = [a and b for a,b in zip(self.action_mask , [False,True,True,True])]
            

        # elif actions == 1:
        #     self.hvacAgent.getState(self.totalState)
        #     self.hvacAgent.execute()
        #     self.updateTotalState("hvac")
        #     self.hvacAgent.rewardStandardization()            
        #     reward.append(self.hvacAgent.reward)
        #     self.action_mask = np.asarray(self.action_mask and [True,False,True,True])

        # elif actions == 2:
        #     self.intAgent.getState(self.totalState)
        #     self.intAgent.execute()
        #     self.updateTotalState("int")
        #     self.intAgent.rewardStandardization()
        #     reward.append(self.intAgent.reward)
        #     self.action_mask = np.asarray(self.action_mask and [True,True,False,True])
        # elif actions == 3:
        #     self.unIntAgent.getState(self.totalState)
        #     self.unIntAgent.execute()
        #     self.updateTotalState("unint")
        #     self.unIntAgent.rewardStandardization()
        #     reward.append(self.unIntAgent.reward)
        #     self.action_mask = np.asarray(self.action_mask and [True,True,True,False])


        reward = sum(reward)
        self.state = self.stateAbstraction(self.totalState)

        #check if all day is done
        done =  bool(sampleTime == 95 and order == 3)

        states = dict(state=self.state,action_mask = self.action_mask)
        print(states)

        return states, done ,reward

    def stateAbstraction(self,totalState) -> np.array:
        # res = []
        # #sampleTime
        # res.append()
        # #SOC
        # res.append()
        # #remain power
        # res.append()
        # #pricePerHour
        # res.append()
        # #HVAC state
        # res.append()
        # #interruptible load state
        # res.append()
        # #uninterruptible load state
        # res.append()
        # res.append()
        return np.array([totalState['sampleTime'],totalState['SOC'],totalState['fixLoad']+totalState['PV']+totalState['deltaSoc']*self.batteryCapacity,totalState['pricePerHour'],True if totalState['userSetTemperature']< totalState['indoorTemperature'] else False,True if totalState['intRemain']==0 else False,True if totalState['unintRemain']==0 else False,totalState['order']],dtype=np.float32)

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
        #Order = 0,1,2,3
        self.totalState["order"] = (self.totalState["order"]+1 if self.totalState["order"]<3 else 0 )
        #SampleTime
        if self.totalState["order"] == 0:
            self.totalState["sampleTime"]+=1
            self.totalState["fixLoad"]=self.Load[self.totalState["sampleTime"]]
            self.totalState["PV"]=self.PV[self.totalState["sampleTime"]]
            self.totalState["pricePerHour"]=self.GridPrice[self.totalState["sampleTime"]]
            self.totalState["outdoorTemperature"]=self.outdoorTemperature[self.totalState["sampleTime"]]
            self.totalState["userSetTemperature"]:self.userSetTemperature[self.totalState["sampleTime"]]

        