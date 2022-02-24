
from tensorforce.execution import Runner  


# agent1 = dict(
#         agent='dqn',
#         # Automatically configured network
#         network='auto',
#         # PPO optimization parameters
#         batch_size=96, update_frequency=5, learning_rate=1e-5, 
#         # Reward estimation
#         discount=0.99, predict_terminal_values=False,
#         reward_processing=None,
#         # Regularization
#         l2_regularization=0.0, entropy_regularization=0.0,
#         # Preprocessing
#         state_preprocessing='linear_normalization',
#         memory=288,
#         # Exploration
#         exploration=0.2, variable_noise=0.0,
#         # Default additional config values
#         config=None,
#         # Save agent every 10 updates and keep the 5 most recent checkpoints
#         saver=dict(directory='saver_dir', load=False, frequency=50),
#         summarizer=dict(directory='summaries',
#         # list of labels, or 'all'
#         summaries='all'),       
#         # Do not record agent-environment interaction trace
#         recorder=None
#     )

# agent = dict(
#         agent='ppo',
#         # Automatically configured network
#         network='auto',
#         # PPO optimization parameters
#         batch_size=10, update_frequency=2, learning_rate=3e-4, multi_step=10,
#         subsampling_fraction=0.33,
#         # Reward estimation
#         likelihood_ratio_clipping=0.2, discount=0.99, predict_terminal_values=False,
#         reward_processing=None,
#         # Baseline network and optimizer
#         baseline=dict(type='auto', size=32, depth=1),
#         baseline_optimizer=dict(optimizer='adam', learning_rate=1e-3, multi_step=10),
#         # Regularization
#         l2_regularization=0.0, entropy_regularization=0.0,
#         # Preprocessing
#         state_preprocessing='linear_normalization',
#         # Exploration
#         exploration=0.0, variable_noise=0.0,
#         # Default additional config values
#         config=None,
#         # Save agent every 10 updates and keep the 5 most recent checkpoints
#         saver=dict(directory='saver_dir', load=False, frequency=50),
#         summarizer=dict(directory='summaries',
#         # list of labels, or 'all'
#         summaries='all'),       
#         # Do not record agent-environment interaction trace
#         recorder=None
#     )


runner = Runner(
    agent= 'agent/dqn.json',
    environment=dict(environment='gym', level='Hems-v0'),
    max_episode_timesteps=96,
)

runner.run(num_episodes=1000,save_best_agent='bestAgent')
runner.agent.save(directory='saved-model', format='saved-model')
runner.close()