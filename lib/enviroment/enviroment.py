from  gym import Env
from  gym.spaces import Discete,Box
import numpy as np

class CemsEnvClass(Env):
    def __init__(self,info) :
        #Actions we take
        # 0. grid+pv+bat->load; 1. grid+bat->load; 2.grid+pv->load; 3.grid->load; 4.grid+pv->bat+load; 5.grid->bat+load
        self.info = info
        self.action_space = Discete(6)
        self.all_day_long = 96
        self.price = []
        self.battery_capacity = 10000
        self.state = {'sample time':0,'grid price': self.info['grid price'][0],'PV': self.info['PV'][0],'SOC':0.2,'sum of load':self.info['sum of load'][0]}
    
    def step(self,action):
        #STATE
        self.all_day_long -=1
        self.state['sample time']+=1
            #get grid price
        self.state['grid price']=self.info['grid price'][self.state['sample time']]
        self.state['PV'] = self.info['PV'][self.state['sample time']]
        self.state['sum of load'] = self.info['sum of load'][self.state['sample time']]
        #ACTION 
            # 0. grid+pv+bat->load
        if action == 0:
            self.state['SOC']-=0.1
            self.price.append((self.state['sum of load']-self.battery_capacity-self.state['PV'])*self.state['grid price'])
            
            # 1. grid+bat->load
        elif action ==1:
            self.state['SOC']-=0.1
            self.price.append((self.state['sum of load']-self.battery_capacity)*self.state['grid price'])
       
            # 2.grid+pv->load
        elif action ==2:
            self.price.append((self.state['sum of load']-self.state['PV'])*self.state['grid price'])
        
            # 3.grid->load
        elif action ==3:
            self.price.append(self.state['sum of load']*self.state['grid price'])
        
            # 4.grid+pv->bat+load
        elif action == 4:
            self.state['SOC']+=0.1
            self.price.append((self.state['sum of load']+self.battery_capacity-self.state['PV'])*self.state['grid price'])
        
            # 5.grid->bat+load
        elif action == 5:
            self.state['SOC']+=0.1
            self.price.append((self.state['sum of load']+self.battery_capacity)*self.state['grid price'])


            #check if all day is done
        if self.all_day_long <= 0:
            done = True
        else:
            done = False



        #REWARD
        if done ==True :
            reward = sum(self.price)
        else:
            reward = 0
        #set placeholder for infomation
        info = {}
        return self.state,reward,done,info

        
    def render(self):
        pass
    def reset(self):
        pass
