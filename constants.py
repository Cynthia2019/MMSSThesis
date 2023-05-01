import numpy as np
env_params = {
    "lookback": 1, 
    "norm_cash": 2 ** -12, 
    "norm_stocks": 2 ** -8, 
    "norm_tech": 2 ** -15, 
    "norm_reward": 2 ** -10, 
    "norm_action": 100
}
erl_params = {
    "learning_rate": 7.5e-3, 
    'batch_size': 512, 
    'gamma': 0.95, 
    'net_dimension': 2**10, 
    'target_step': 5e4, 
    'eval_time_gap': 60, 
    'break_step': 4.5e4
}

ALPACA_LIMITS = np.array([0.01,
                        0.0001,
                        0.001,
                        0.01,
                        0.01,
                        0.05,
                        ])


A2C_model_kwargs = {
                    'n_steps': 5,
                    'ent_coef': 0.005,
                    'learning_rate': 0.0007
                    }

PPO_model_kwargs = {
                    "ent_coef":0.01,
                    "n_steps": 2, #2048
                    "learning_rate": 7.5e-3,
                    "batch_size": 512
                    }

DDPG_model_kwargs = {
                      #"action_noise":"ornstein_uhlenbeck",
                      "buffer_size": 1, #10_000
                      "learning_rate": 0.0005,
                      "batch_size": 64
                    }

timesteps_dict = {'a2c' : 1, #10_000 each
                 'ppo' : 1, 
                 'ddpg' : 1
                 }