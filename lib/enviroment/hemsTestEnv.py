from  gym.envs.Hems.import_data import ImportData 
from  gym import Env
from  gym import spaces
from gym import make
import numpy as np
from  yaml import load , SafeLoader

class HemsEnv(Env):
    def __init__(self) :
        '''
        Action space
        observation space
        '''
        #
        # The information of ip should   'NOT'   upload to github
        #
        with open("yaml/mysqlData.yaml","r") as f:
            self.mysqlData = load(f,SafeLoader)

        self.host = self.mysqlData['host']
        self.user = self.mysqlData['user']
        self.passwd = self.mysqlData['passwd']
        self.db = self.mysqlData['db']
        self.info = ImportData(host= self.host ,user= self.user ,passwd= self.passwd ,db= self.db,mode = 'Testing')
        self.BaseParameter = self.info.experimentData['BaseParameter']
        self.GridPrice = self.info.experimentData['GridPrice']['summer_price'].tolist()

        #each month pick one day for testing
        self.i = 0
        self.Load = self.info.experimentData['Load'].iloc[:,self.i].tolist()
        if self.i % 12 == 0:
            self.PV = self.info.experimentData['PV']['Jan'].tolist()
        elif self.i % 12 == 1:
            self.PV = self.info.experimentData['PV']['Feb'].tolist()
        elif self.i % 12 == 2:
            self.PV = self.info.experimentData['PV']['Mar'].tolist()
        elif self.i % 12 == 3:
            self.PV = self.info.experimentData['PV']['Apr'].tolist()
        elif self.i % 12 == 4:
            self.PV = self.info.experimentData['PV']['May'].tolist()
        elif self.i % 12 == 5:
            self.PV = self.info.experimentData['PV']['Jun'].tolist()
        elif self.i % 12 == 6:
            self.PV = self.info.experimentData['PV']['July'].tolist()
        elif self.i % 12 == 7:
            self.PV = self.info.experimentData['PV']['Aug'].tolist()
        elif self.i % 12 == 8:
            self.PV = self.info.experimentData['PV']['Sep'].tolist()
        elif self.i % 12 == 9:
            self.PV = self.info.experimentData['PV']['Oct'].tolist()
        elif self.i % 12 == 10:
            self.PV = self.info.experimentData['PV']['Nov'].tolist()
        elif self.i % 12 == 11:
            self.PV = self.info.experimentData['PV']['Dec'].tolist()
        
        #action we take (degree of charging/discharging power)
        self.action_space = spaces.Box(low=-0.025,high=0.025,shape=(1,),dtype=np.float32)

        #observation space ( Only SOC matters )
        self.observation_space_name = np.array(['sampleTime', 'load', 'pv', 'SOC', 'pricePerHour'])
        upperLimit = np.array(
            [
                #timeblock
                96,
                #load
                np.finfo(np.float32).max,
                #PV
                np.finfo(np.float32).max,
                #SOC
                self.BaseParameter.loc[self.BaseParameter['parameter_name']=='SOCmax','value'],
                #pricePerHour
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        lowerLimit = np.array(
            [
                #timeblock
                0,
                #load
                np.finfo(np.float32).min,
                #PV
                np.finfo(np.float32).min,
                #SOC
                self.BaseParameter.loc[self.BaseParameter['parameter_name']=='SOCmin','value'],         
                #pricePerHour
                np.finfo(np.float32).min,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(lowerLimit,upperLimit,dtype=np.float32)
        self.state = None
        
    def step(self,action):
        '''
        interaction of each state(changes while taking action)
        Rewards
        Episode Termination condition
        '''
        #error message if getting wrong action
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action),err_msg

        #STATE (sampleTime,Load,PV,SOC,pricePerHour)
        sampleTime,load,pv,soc,pricePerHour = self.state
        soc_change = float(action)
        # action(soc_change) is the degree of charging/discharging power .
        # if soc_change > 0 means charging , whereas soc_change<0 means discharging.

        #interaction
        # if energy supply is greater than consumption
        if (pv + soc_change*float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='batteryCapacity']['value'])[0])) >= load :
            cost = 0.001

        
        # if energy supply is less than consumption
        else:
            if (soc + soc_change) < 0 :
                soc = 0
                cost = 0.001
            elif (soc + soc_change) > 1:
                soc = 1
                cost = 0.001
            else:
            #calculate the new soc for next state
                soc = soc+soc_change
                #calculate the cost at this sampletime (multiple 0.25 is for transforming pricePerHour  into per 15 min)
                cost = pricePerHour * 0.25 *( load + soc_change*float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='batteryCapacity']['value'])[0]) - pv  )


            


        #change to next state
        sampleTime = int(sampleTime+1)
        #self.state = {'sampleTime':sampleTime,'load': self.Load[self.state['sampleTime']],'pv': self.PV[self.state['sampleTime']],'SOC':soc,'pricePerHour':self.GridPrice[self.state['sampleTime']]}
        self.state=np.array([sampleTime,self.Load[sampleTime],self.PV[sampleTime],soc,self.GridPrice[sampleTime]])
        #check if all day is done
        done = bool(
            sampleTime == 95
        )

        #REWARD
        reward = []
        if not done:
            #punish if the agent choose the action which shouldn't be choose(charge when SOC is full or discharge when SOC is null)
            if (soc >= 1 and soc_change > 0) or (soc <= 0 and soc_change < 0) :
                reward.append(-2)
            # reward 1
            r1 = -cost/10000*1.08
            reward.append(r1)
            #reward 2
            if cost / (pricePerHour*0.25) >= 20000:
                reward.append(-5)
                if soc_change < 0:
                    reward.append(2)
            else:    
                reward.append(0.0625)
            reward.append(soc- float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='SOCthreshold']['value'])[0]))


        # if done
        else : 
            if (soc >= 1 and soc_change > 0) or (soc <= 0 and soc_change < 1) :
                reward.append(-2)
            # reward 1
            r1 = -cost/10000*1.08
            reward.append(r1)
            #reward 2
            if cost / (pricePerHour*0.25) >= 20000:
                reward.append(-5)
                if soc_change < 1:
                    reward.append(2)
            else:    
                reward.append(0.0625)


            # reward 3
            r2 =  20*(soc - float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='SOCthreshold']['value'])[0]))
            reward.append(r2)


        reward = sum(reward)



        #set placeholder for infomation
        info = {}

        return self.state,reward,done,info

        
    def render(self):
        pass
    def reset(self):
        '''
        Starting State
        '''

        #each month pick one day for testing
        self.i += 1
        self.Load = self.info.experimentData['Load'].iloc[:,self.i]

        if self.i % 12 == 0:
            self.PV = self.info.experimentData['PV']['Jan'].tolist()
        elif self.i % 12 == 1:
            self.PV = self.info.experimentData['PV']['Feb'].tolist()
        elif self.i % 12 == 2:
            self.PV = self.info.experimentData['PV']['Mar'].tolist()
        elif self.i % 12 == 3:
            self.PV = self.info.experimentData['PV']['Apr'].tolist()
        elif self.i % 12 == 4:
            self.PV = self.info.experimentData['PV']['May'].tolist()
        elif self.i % 12 == 5:
            self.PV = self.info.experimentData['PV']['Jun'].tolist()
        elif self.i % 12 == 6:
            self.PV = self.info.experimentData['PV']['July'].tolist()
        elif self.i % 12 == 7:
            self.PV = self.info.experimentData['PV']['Aug'].tolist()
        elif self.i % 12 == 8:
            self.PV = self.info.experimentData['PV']['Sep'].tolist()
        elif self.i % 12 == 9:
            self.PV = self.info.experimentData['PV']['Oct'].tolist()
        elif self.i % 12 == 10:
            self.PV = self.info.experimentData['PV']['Nov'].tolist()
        elif self.i % 12 == 11:
            self.PV = self.info.experimentData['PV']['Dec'].tolist()
        #reset state
        self.state=np.array([0,self.Load[0],self.PV[0],float(list(self.BaseParameter.loc[self.BaseParameter['parameter_name']=='SOCinit']['value'])[0]),self.GridPrice[0]])
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
        