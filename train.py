from tensorforce.execution import Runner  
from tensorforce import Agent
import sys

if __name__ == '__main__':
    if len(sys.argv) >2:    
        if sys.argv[1] == 'pretrain':
            runner = Runner(
                environment=dict(environment='gym', level='Hems-v0'),
                agent= Agent.load(directory = 'saver_dir',format='checkpoint'),
                max_episode_timesteps=96,
            )
    else:
        runner = Runner( 
            environment=dict(environment='gym', level='Hems-v0'),
            agent= 'agent/DDPG.json',
            max_episode_timesteps=96,
        )



    runner.run(num_episodes=1200,save_best_agent='bestAgent')
    runner.close()