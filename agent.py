from __future__ import annotations

import time

import numpy as np
import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

MODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}

NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
}

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        try:
            self.logger.record(key="train/reward", value=self.locals["rewards"][0])
        except BaseException:
            self.logger.record(key="train/reward", value=self.locals["reward"][0])
        return True

# combination between finrl_crypto agent and sb3 agent class 
# Stable Baseline 3 class
class DRLAgent: 
  def __init__(self, env):
      self.env = env

  def get_model(
      self, 
      model_name, 
      policy="MlpPolicy", 
      policy_kwargs=None, 
      model_kwargs=None, 
      verbose=1, 
      seed=None, 
      tensorboard_log=None,
  ):
    if model_name not in MODELS:
        raise NotImplementedError("NotImplementedError")
    
    if "action_noise" in model_kwargs:
      n_actions = self.env.action_space.shape[-1]
      model_kwargs["action_noise"] = NOISE[model_kwargs["action_noise"]](
          mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
      )
    print(model_kwargs)    
    return MODELS[model_name](
                policy=policy,
                env=self.env,
                tensorboard_log=tensorboard_log,
                verbose=verbose,
                policy_kwargs=policy_kwargs,
                seed=seed,
                **model_kwargs,
            )
    

  def train_model(self, model, tb_log_name, total_timesteps=5000, callback=TensorboardCallback()):
      model = model.learn(
          total_timesteps=total_timesteps,
          tb_log_name=tb_log_name,
          callback=callback,
      )
      return model

  @staticmethod
  def DRL_prediction(model, environment, deterministic=True):
      test_env, test_obs = environment.get_sb_env()
      """make a prediction"""
      account_memory = []
      # actions_memory = []
      #         state_memory=[] #add memory pool to store states
      test_env.reset()
      for i in range(len(environment.df.index.unique())):
          action, _states = model.predict(test_obs, deterministic=deterministic)
          test_obs, rewards, dones, info = test_env.step(action)
          account_memory = test_env.env_method(method_name="save_asset_memory")
        #   if i == (len(environment.df.index.unique()) - 2):
        #       account_memory = test_env.env_method(method_name="save_asset_memory")
          if dones[0]:
              print("hit end!", i)
              print("rewards", rewards)
              break
      return account_memory