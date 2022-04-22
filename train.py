from tensorforce.execution import Runner  

runner = Runner(
    agent= 'loadAgent/D3qn.json',
    environment=dict(environment='gym', level='Hems-v4'),
    max_episode_timesteps=96,
)

runner.run(num_episodes=540,save_best_agent='bestAgent')
runner.close()