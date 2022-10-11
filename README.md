# RL Environment for Home Energy Management system

## Introduction

- A RL environment Used for Hems which based on *OpenAI Gym*
- Build the RL Agent with *Tensorforce*

## Brief view of the project
-  train.py is for training RL agent in environment "hemsTrainEnv.py" which locate at lib/enviroment
-  test.py is for testing the agent we've already trained in a different env "hemsTestEnv.py" , and also output a result graph in "pic", which shows the SOC and price in each test.
-  Difference between  "hemsTrainEnv.py" and "hemsTestEnv.py" is the import data. The previous one grab 360 days load consumption from mysql , while the other one picks up only 12 days for testing.
-  "agent" stores the json file of hyperparameter for building different kind of RL agents.
- Using my own mysql Database for the training and testing data , "import_data.py" is for grabbing them from the server.

## Env
### single agent Env
- SOC system
- HVAC system
- Interruptible load system
- Uninterruptible load system
### multi agent Env
- SOC and HVAC system
## details in hemsTrainEnv
- __init__()
    1. import experiment parameters , training dataset from mysql database
    1. set the Observation space

    | name | upper Limit   | lower Limit |
    | --- |    ---    | --- |
    | sampletime    |     96       |   0    | 
    | load consumption   |     infinity       | -infinity      | 
    | photovoltaic    |    infinity        |   -infinity    | 
    | SOC    |     1     |   0    | 
    | pricePerHour    |   infinity         |  -infinity     |
    3. set the Action space
    which is a Discrete space of three actions "charge" , "discharge" , and "stay".  
- step(action)
    1. step() is literally for go through one step in the env . Which returns you the next state , the reward you get from this state (through the action you feed in) , whether the whole episode is done or not (96 time step for each episode) .
    2. the mathematical formula for building reward functions, please refer to https://app.heptabase.com/w/16a6f10f040b1a8ec108a0ce0f1a13556489be7c8a6885489f4f7489e4baf789
- reset()
    1. For initializing the Env into $S_0$
    2. Which will randomly pick a training dataset from 360 days 


    
