import gym
import random
import time

from gym import envs
env = gym.make("Hems-v0")
state = env.reset() # 在第一次step前要先重置環境 不然會報錯
print('state: ',state)
action_space = env.action_space
while True:
    action = action_space.sample() # 隨機動作
    print('action: ',action)
    state, reward, done, info = env.step(action)
    print('state: ',state)
    print('reward: ',reward)
    
    if done: break