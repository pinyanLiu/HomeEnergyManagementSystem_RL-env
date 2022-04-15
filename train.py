from tensorforce.execution import Runner  

runner = Runner(
    agent= 'agent/DDPG.json',
    environment=dict(environment='gym', level='Hems-v0'),
    max_episode_timesteps=96,
)

runner.run(num_episodes=10000,save_best_agent='bestAgent')
runner.close()