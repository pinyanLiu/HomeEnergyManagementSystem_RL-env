from tensorforce.execution import Runner  
import sys
from lib.enviroment.UnInterruptibleLoadTrainEnv import UnIntEnv
from lib.enviroment.InterruptibleLoadTrainEnv import IntEnv
from lib.enviroment.SocTrainEnv import SocEnv
from lib.enviroment.HVACTrainEnv import HvacEnv
from lib.enviroment.multiAgentEnv.multiAgentTrainEnv import multiAgentTrainEnv
def main(argv):
    if len(argv)<4:
        print('please give parameters 1.Training mode: "soc"/"intload"/"HVAC"/"unintload"/"hla" 2.Agent 3. training episode ')
        return
    agent = argv[2]

    if argv[1] == 'soc':
        environment = dict(environment=SocEnv)
        bestAgentDir = 'Soc/bestAgent/'
    elif argv[1] == 'int':
        environment = dict(environment=IntEnv)
        bestAgentDir = 'Load/Interruptible/bestAgent/'
    elif argv[1] == 'hvac':    
        environment = dict(environment=HvacEnv)
        bestAgentDir = 'Load/HVAC/bestAgent/'
    elif argv[1] == 'unint':
        environment = dict(environment=UnIntEnv)
        bestAgentDir = 'Load/Uninterruptible/bestAgent/'
    elif argv[1] == 'hla':
        environment = dict(environment = multiAgentTrainEnv)
        bestAgentDir = 'Load/HLA/bestAgent/'
        
    else :
        print('please give parameters 1.Training mode: "soc"/"intload"/"hvac"/"unintload"/"hla" 2.Agent 3. training episode ')
        return
    if argv[1]== 'hla':
        runner = Runner(
            environment=environment,
            agent= agent,
            max_episode_timesteps=960,
        )
    else:
        runner = Runner(
            environment=environment,
            agent= agent,
            max_episode_timesteps=96,
        )


    
    runner.run(num_episodes=int(argv[3]),save_best_agent=bestAgentDir)
    runner.close()


if __name__ == '__main__':
    main(sys.argv)
