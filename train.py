from tensorforce.execution import Runner  

runner = Runner(
    agent= 'agent/dqn.json',
    environment=dict(environment='gym', level='Hems-v0'),
    max_episode_timesteps=96,
)

runner.run(num_episodes=20000,save_best_agent='bestAgent')
runner.close()