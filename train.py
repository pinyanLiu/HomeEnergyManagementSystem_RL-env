from ast import arg
from tensorforce.execution import Runner  
from tensorforce import Agent
import sys

def main(argv):
    if len(argv)<4:
        print('please give parameters 1.Training mode: "soc"or"load" 2.Pretrain:"True","False" 3. training episode ')
        return

    if argv[1] == 'soc':
        environment = dict(environment='gym', level='Hems-v0')
    elif argv[1] == 'load':
        environment = dict(environment='gym', level='Hems-v4')
    else :
        print('undefined env')
        return
    
    if argv[2] == 'True':
        agent = Agent.load(directory = 'Pretrain_dir',format='checkpoint')
    elif argv[2] == 'False':
        agent = 'loadAgent/D3qn.json'
    else :
        print('undefined agent')
        return

    runner = Runner(
        environment=environment,
        agent= agent,
        max_episode_timesteps=96,
    )

    runner.run(num_episodes=int(argv[3]),save_best_agent=True)
    runner.close()


if __name__ == '__main__':
    main(sys.argv)
