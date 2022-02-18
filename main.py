
from posixpath import basename
from tensorforce.execution import Runner  
from tensorforce.environments import  Environment
from tensorforce.agents import Agent 
#from gym.envs.enviroment.enviroment import HemsEnv

#env = gym.make("Hems-v0")
# environment = Environment.create(environment='gym',level = 'Hems-v0', max_episode_timesteps=96)
# agent = Agent.create(agent = 'agent.json',environment = environment)

# for ep in range(500): # Number of episodes

#     print('********Episode ' + str(ep) + '********')

#     # Initialize episode
#     states = environment.reset()
#     done = False
#     step = 0

#     while not done: # Episode timestep
#         actions = agent.act(states=states)
#         states, done, reward = environment.execute(actions=actions)
#         agent.observe(terminal=done, reward=reward)
#         #environment.render() # Gives error

# environment.close()
# agent.close()

agent = dict(
        agent='ppo',
        # Automatically configured network
        network='auto',
        # PPO optimization parameters
        batch_size=10, update_frequency=2, learning_rate=3e-4, multi_step=10,
        subsampling_fraction=0.33,
        # Reward estimation
        likelihood_ratio_clipping=0.2, discount=0.99, predict_terminal_values=False,
        reward_processing=None,
        # Baseline network and optimizer
        baseline=dict(type='auto', size=32, depth=1),
        baseline_optimizer=dict(optimizer='adam', learning_rate=1e-3, multi_step=10),
        # Regularization
        l2_regularization=0.0, entropy_regularization=0.0,
        # Preprocessing
        state_preprocessing='linear_normalization',
        # Exploration
        exploration=0.0, variable_noise=0.0,
        # Default additional config values
        config=None,
        # Save agent every 10 updates and keep the 5 most recent checkpoints
        saver=dict(directory='saver_dir', load=False, frequency=50),
        summarizer=dict(directory='summaries',
        # list of labels, or 'all'
        summaries='all'),       
        # Do not record agent-environment interaction trace
        recorder=None
    )

runner = Runner(
    agent=agent,
    environment=dict(environment='gym', level='Hems-v0'),
    max_episode_timesteps=30000,
)

runner.run(num_episodes=29999)
runner.agent.save(directory='saved-model', format='saved-model')
runner.close()