import gym

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from constants import A2C_model_kwargs, PPO_model_kwargs, timesteps_dict, DDPG_model_kwargs, erl_params, env_params

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from load_data import load_data, preprocess_data, df_to_array, dump_data
from feature_engineer import data_split
from CryptoEnv import CryptoEnv
from CryptoEnvTurbulence import CryptoEnvTurbulence
from agent import DRLAgent

import tensorflow as tf
print(f"{tf.__version__=}")
print(f"{gym.__version__=}")

ALPACA_LIMITS = np.array([0.01,
                        0.0001,
                        0.001,
                        0.01,
                        0.01,
                        0.05,
                        ])



def main(): 
    print("Loading data...")
    train_data = load_data('data/train_data')
    trade_data = load_data('data/trade_data')
    train_data = preprocess_data(train_data)
    trade_data = preprocess_data(trade_data)

    train_data = train_data.drop(columns=['CVIX'])

    TRAIN_START_DATE = train_data.loc[0, 'date']
    TRAIN_END_DATE = train_data.loc[train_data.shape[0] - 1, 'date']
    TRADE_START_DATE = trade_data.loc[0, 'date']
    TRADE_END_DATE = trade_data.loc[trade_data.shape[0] - 1, 'date']

    train_data = data_split(train_data, TRAIN_START_DATE, TRAIN_END_DATE)
    trade_data = data_split(trade_data, TRADE_START_DATE, TRADE_END_DATE)

    train_price_array, train_tech_array, train_time_array = df_to_array(train_data)
    trade_price_array, trade_tech_array, trade_time_array = df_to_array(trade_data)

    dump_data(train_data, 'processed/train_data')
    dump_data(trade_data, 'processed/trade_data')

    crypto_indicators = train_data.columns.values
    crypto_indicators = np.setdiff1d(crypto_indicators, ['date', 'close', 'tic', 'turbulence'])

    insample_turbulence = train_data[
            (train_data.date < TRAIN_END_DATE)
            & (train_data.date >= TRAIN_START_DATE)
        ]
    insample_turbulence_threshold = np.quantile(
        insample_turbulence.turbulence.values, 0.95
    )


    models = {
        "a2c": A2C_model_kwargs,
        "ppo": PPO_model_kwargs,
        "ddpg": DDPG_model_kwargs,
    }
    for model_name, model_kwargs in models.items(): 
        
        print("Creating environment...")
        train_env = CryptoEnvTurbulence(
            config={
                "price_array": train_price_array,
                "tech_array": train_tech_array,
                "df": train_data,
            }, 
            env_params=env_params,
            tech_indicator_list=crypto_indicators,
            mode="train",
            model_name=model_name,
            turbulence_threshold=insample_turbulence_threshold
        )

        trade_env = CryptoEnvTurbulence(
            config={
                "price_array": trade_price_array,
                "tech_array": trade_tech_array,
                "df": trade_data,
            },
            env_params=env_params,
            tech_indicator_list=crypto_indicators,
            mode="trade",
            model_name=model_name, 
            turbulence_threshold=insample_turbulence_threshold
        )

        print("Creating agent...")
        agent = DRLAgent(DummyVecEnv([lambda: train_env]))
        model = agent.get_model(model_name, model_kwargs=model_kwargs)
        print("Training...")
        model = agent.train_model(model, f"{model_name}_train", total_timesteps=10)

        print("Trading...")
        agent.DRL_prediction(model, trade_env)



if __name__ == "__main__":
    main()
