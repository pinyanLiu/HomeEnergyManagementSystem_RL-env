from ast import arg
from tensorforce.execution import Runner  
from tensorforce import Agent
import sys

def main(argv):
    if len(argv)<4:
        print('please give parameters 1.Training mode: "soc"or"load" 2.Agent 3. training episode ')
        return
    agent = argv[2]

    if argv[1] == 'soc':
        environment = dict(environment='gym', level='Hems-v0')

    elif argv[1] == 'load':
        environment = dict(environment='gym', level='Hems-v4')
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
