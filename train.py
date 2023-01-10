from tensorforce.execution import Runner  
import sys
from lib.enviroment.UnInterruptibleLoadTrainEnv import UnIntEnv

def main(argv):
    if len(argv)<4:
        print('please give parameters 1.Training mode: "soc"/"intload"/"HVAC"/"unintload 2.Agent 3. training episode ')
        return
    agent = argv[2]

    if argv[1] == 'soc':
        environment = dict(environment='gym', level='Hems-v0')
    elif argv[1] == 'intload':
        environment = dict(environment='gym', level='Hems-v4')
    elif argv[1] == 'HVAC':    
        environment = dict(environment='gym', level='Hems-v6')
    elif argv[1] == 'unintload':
        environment = dict(environment=UnIntEnv)
    else :
        print('undefined env')
        return
    
    runner = Runner(
        environment=environment,
        agent= agent,
        max_episode_timesteps=96,
    )


    
    runner.run(num_episodes=int(argv[3]),save_best_agent='bestAgent')
    runner.close()


if __name__ == '__main__':
    main(sys.argv)
