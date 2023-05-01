import numpy as np
import pandas as pd
import requests
import gym
import json, prettyprint, copy
import collections, math, time
from datetime import datetime, timezone, timedelta
import os
import matplotlib.pyplot as plt
from constants import *
from gym import spaces
from gym.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv

class CryptoEnvTurbulence(gym.Env):

    metadata = {"render.modes": ["rgb_array"]}
    def __init__(self, config, env_params, window_size=1, initial_capital=10000,
                 buy_cost_pct=0.0025, sell_cost_pct=0.0015, gamma=0.95, if_log=False, 
                 print_verbosity=3, save_result=True, initial=True, turbulence_threshold=None, 
                 previous_state=[], tech_indicator_list=[], 
                 model_name="", mode="", iteration=""):
        self.if_log = if_log
        self.env_params = env_params
        self.lookback = env_params['lookback']
        self.initial_total_asset = initial_capital
        self.initial_cash = initial_capital
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.gamma = gamma
        self.window_size = window_size
        self.print_verbosity = print_verbosity
        self.save_result = save_result
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        self.initial = initial 
        self.previous_state = previous_state
        self.tech_indicator_list = tech_indicator_list
        self.turbulence_threshold = turbulence_threshold

        # Get initial price array to compute eqw
        self.price_array = config['price_array']
        self.prices_initial = list(self.price_array[0, :])
        self.equal_weight_stock = np.array([self.initial_cash /
                                            len(self.prices_initial) /
                                            self.prices_initial[i] for i in
                                            range(len(self.prices_initial))])
        
        # read normalization of cash, stocks and tech
        self.norm_cash = env_params['norm_cash']
        self.norm_stocks = env_params['norm_stocks']
        self.norm_tech = env_params['norm_tech']
        self.norm_reward = env_params['norm_reward']
        self.norm_action = env_params['norm_action']

        # Initialize constants
        self.tech_array = config['tech_array']
        self._generate_action_normalizer()
        self.crypto_num = self.price_array.shape[1]
        self.max_step = self.price_array.shape[0] - self.lookback - 1

        # reset
        # time is an index for timestamp/date
        self.time = self.lookback - 1
        self.cash = self.initial_cash
        self.current_price = self.price_array[self.time]
        self.current_tech = self.tech_array[self.time]
        self.stocks = np.zeros(self.crypto_num, dtype=np.float32) #num stock shares 
        self.stocks_cooldown = None
        self.safety_factor_stock_buy = 1 - 0.1

        # equivalent: self.initial_amount
                # + np.sum(
                #     np.array(self.num_stock_shares)
                #     * np.array(self.state[1 : 1 + self.stock_dim])
                # )
        self.total_asset = self.cash + (self.stocks * self.price_array[self.time]).sum()
        self.total_asset_eqw = np.sum(self.equal_weight_stock * self.price_array[self.time])

        self.episode_return = 0.0
        self.gamma_return = 0.0
        self.episode = 0

        '''env information'''
        self.env_name = 'MulticryptoEnv'

        # state_dim = cash[1,1] + stocks[1,4] + tech_array[1,44] * lookback + stock_cooldown[1,4]
        self.state_dim = 1 + self.price_array.shape[1] + self.crypto_num + self.tech_array.shape[1] * self.lookback
        self.action_dim = self.price_array.shape[1]
        self.minimum_qty_alpaca = ALPACA_LIMITS * 1.1  # 10 % safety factor
        self.if_discrete = False

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,))

        self.df = config['df']
        self.data = self.df.loc[self.time, :] #same as self.day in stocktradingenv

        self.state = self._initiate_state()

        # initial state memory, actions memory, rewards memory
        self.asset_memory = [self.total_asset]
        self.actions_memory = []
        self.state_memory = ([])
        self.date_memory = [self._get_date()]
        self.price_memory = []
        self.stocks_memory = []

        self._seed()

    def reset(self) -> np.ndarray:


        if self.initial: 
            self.total_asset = self.cash + (self.stocks * self.price_array[self.time]).sum()
            self.asset_memory = [self.total_asset]
            self.cash = self.initial_cash
            self.stocks = np.zeros(self.crypto_num, dtype=np.float32)
            # self.asset_memory = [self.initial_cash + (self.stocks * self.price_array[self.time]).sum()]
        else: 
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.previous_state[1 : (self.crypto_num + 1)]) * 
                np.array(self.previous_state[(self.crypto_num + 1) : (self.crypto_num * 2 + 1)]) 
            )
            self.asset_memory = [previous_total_asset]
            self.stocks = self.previous_state[(self.crypto_num + 1) : (self.crypto_num * 2 + 1)]
            # self.total_asset = self.cash + (self.stocks * self.price_array[self.time]).sum()
            self.total_asset = previous_total_asset
            self.cash = self.previous_state[0]

        self.state = self._initiate_state()
        self.time = self.lookback - 1
        self.current_price = self.price_array[self.time]
        self.current_tech = self.tech_array[self.time]
        # self.cash = self.initial_cash  # reset()
        # self.stocks = np.zeros(self.crypto_num, dtype=np.float32)
        self.stocks_cooldown = np.zeros_like(self.stocks)
        
        self.data = self.df.loc[self.time, :]

        self.turbulence = 0
        self.episode += 1

        #reset memory 
        self.date_memory = [self._get_date()]
        self.actions_memory = []
        self.state_memory = []
        return self.state

    def step(self, actions):
        done = self.time == self.max_step

        # assert self.price_array[self.time].tolist() == self.data.close.values.tolist(), "price part not equal"
        # assert self.stocks.tolist() == self.state[(self.crypto_num + 1) : (self.crypto_num * 2 + 1)].tolist(), "stocks part not equal"

        # done = self.time >= len(self.df.index.unique()) - 1
        if done:
            reward = self.gamma_return
            self.episode_return = self.total_asset / self.initial_cash
            next_total_asset = self.cash + (self.stocks * self.price_array[self.time]).sum()

            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            if self.episode % self.print_verbosity == 0: 
              print(f"time: {self.time}, episode: {self.episode}")
              print(f"begin_total_asset: {self.total_asset:0.2f}")
              print(f"end_total_asset: {next_total_asset:0.2f}")
              print(f"reward: {reward:0.2f}")
              print("=================================")
            
            if (self.save_result):
                result_folder = 'results'
                if not os.path.exists(result_folder):
                  os.makedirs(result_folder)
                df_actions = self.save_action_memory()
                df_actions.to_csv(
                    "results/actions_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    )
                )
                df_total_value.to_csv(
                    "results/account_value_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                plt.plot(self.asset_memory, "r")
                plt.savefig(
                    "results/account_value_{}_{}_{}.png".format(
                        self.mode, self.model_name, self.iteration
                    )
                )
                plt.close()

            print("episode_return: ", self.episode_return - 1, '\n')
        else:  
          # if a stock is held add to its cooldown
            for i in range(len(actions)):
                if self.stocks[i] > 0:
                    self.stocks_cooldown[i] += 1

            price = self.price_array[self.time]
            for i in range(self.action_dim):
                norm_vector_i = self.action_norm_vector[i]
                actions[i] = round(actions[i] * norm_vector_i, 9)
        
            # Compute actions in dollars
            #   actions_dollars = actions * price

            # Sell
            #######################################################################################################
            #######################################################################################################
            #######################################################################################################
            if self.turbulence < self.turbulence_threshold:
                # normal sell logic
                for index in np.where(actions < -self.minimum_qty_alpaca)[0]:
                    if self.stocks[index] > 0:
                        if price[index] > 0:  # Sell only if current asset is > 0
                            sell_num_shares = min(self.stocks[index], -actions[index])
                            assert sell_num_shares >= 0, "Negative sell!"
                            self.stocks_cooldown[index] = 0
                            self.stocks[index] -= sell_num_shares # self.cash == self.state[0]
                            self.cash += price[index] * sell_num_shares * (1 - self.sell_cost_pct)
                #   # FORCE 5% SELL every half day (30 min timeframe -> (24 * 2 / 2) * 30)
                # 5m timeframe -> (12 * 60 / 5) = 144 
                for index in np.where(self.stocks_cooldown >= 144)[0]:
                    sell_num_shares = self.stocks[index] * 0.05
                    self.stocks_cooldown[index] = 0
                    self.stocks[index] -= sell_num_shares
                    self.cash += price[index] * sell_num_shares * (1 - self.sell_cost_pct)

            else:
                # turbulence regime: sell all holding
                for index in np.where(actions < -self.minimum_qty_alpaca)[0]:
                    if self.stocks[index] > 0: 
                        if price[index] > 0: 
                            sell_num_shares = self.stocks[index]
                            sell_amount = price[index] * sell_num_shares * (1 - self.sell_cost_pct)
                            self.stocks[index] = 0
                            self.cash += sell_amount
            # Buy
            #######################################################################################################
            #######################################################################################################
            #######################################################################################################
            if self.turbulence < self.turbulence_threshold:
                for index in np.where(actions > self.minimum_qty_alpaca)[0]:
                    # print(price, index, actions)
                    if price[index] > 0:  # Buy only if the price is > 0 (no missing data in this particular date)

                        fee_corrected_asset = self.cash / (1 + self.buy_cost_pct)
                        max_stocks_can_buy = (fee_corrected_asset / price[index]) * self.safety_factor_stock_buy
                        buy_num_shares = min(max_stocks_can_buy, actions[index])
                        if buy_num_shares < self.minimum_qty_alpaca[index]:
                            buy_num_shares = 0
                        self.stocks[index] += buy_num_shares
                        self.cash -= price[index] * buy_num_shares * (1 + self.buy_cost_pct)
            else: 
                buy_num_shares = 0

            if self.turbulence >= self.turbulence_threshold:
                actions = np.array([-1 * stock for stock in self.stocks] )


            self.actions_memory.append(actions)

            """update time"""
            self.time += 1
            self.data = self.df.loc[self.time, :]
            self.turbulence = self.data['turbulence'].values[0]
            # done = self.time == self.max_step

            self.state = self._update_state()
            
            next_total_asset = self.cash + (self.stocks * self.price_array[self.time]).sum()
            next_total_asset_eqw = np.sum(self.equal_weight_stock * self.price_array[self.time])


            # Difference in portfolio value + cooldown management
            delta_bot = next_total_asset - self.total_asset
            delta_eqw = next_total_asset_eqw - self.total_asset_eqw

            # Reward function
            reward = (delta_bot - delta_eqw) * self.norm_reward
            self.total_asset = next_total_asset
            self.total_asset_eqw = next_total_asset_eqw

            self.gamma_return = self.gamma_return * self.gamma + reward
            self.cumu_return = self.total_asset / self.initial_cash

            # update memory 
            self.asset_memory.append(next_total_asset)
            self.date_memory.append(self._get_date())
            self.state_memory.append(self.state)

        return self.state, reward, done, {}

    def render(self, mode="rgb_array"): 
      return self.state
    
    def _get_date(self): 
      if len(self.df.tic.unique()) > 1: 
        date = self.data.date.unique()[0]
      else: 
        date = self.data.date
      return date

    def get_final_state(self): 
      return self.state
    
    
    def save_asset_memory(self): 
      date_list = self.date_memory
      asset_list = self.asset_memory
      df_account_value = pd.DataFrame({"date": date_list, "account_value": asset_list})
      return df_account_value


    def save_action_memory(self): 
      if len(self.df.tic.unique()) > 1:
        date_list = self.date_memory[:-1]
        df_date = pd.DataFrame(date_list)
        df_date.columns = ["date"]

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        # if len(action_list) != len(self.data.tic.values): 
        #   return df_actions
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
      else: 
        date_list = self.date_memory[:-1]
        action_list = self.actions_memory
        df_actions = pd.DataFrame({"date": date_list, "actions": action_list})

      return df_actions


    def _initiate_state(self):
        if self.initial:
            state = (
                [self.cash]
                + self.data.close.values.tolist()
                + list(self.stocks)
                + sum(
                    (
                        list(map(lambda x : x * 1, self.data[tech].values.tolist()))
                        for tech in self.tech_indicator_list
                    ),
                    []
                )
            ) 
        else: 
            state = (
                [self.previous_state[0]]
                + self.data.close.values.tolist()
                + list(self.previous_state[(self.crypto_num + 1): (self.crypto_num * 2 + 1)])
                + sum(
                    (
                        list(map(lambda x : x * 1, self.data[tech].values.tolist()))
                        for tech in self.tech_indicator_list
                    ),
                    []
                )
            ) 
            self.cash = self.previous_state[0]
            self.stocks = self.previous_state[self.crypto_num + 1 : self.crypto_num * 2 + 1]
        
        return state
        # return np.array(state, dtype=np.float32)
    
    def _update_state(self):
        state = (
            [self.cash] 
            + self.data.close.values.tolist()
            + list(self.stocks)
            + sum(
                    (
                        list(map(lambda x : x * 1, self.data[tech].values.tolist()))
                        for tech in self.tech_indicator_list
                    ),
                    []
                )
        )

        return state

        # return np.array(state, dtype=np.float32)

    def close(self):
        pass

    def _generate_action_normalizer(self):
        action_norm_vector = []
        price_0 = self.price_array[0]
        for price in price_0:
            x = math.floor(math.log(price, 10))  # the order of magnitude
            action_norm_vector.append(1 / ((10) ** x))

        action_norm_vector = np.asarray(action_norm_vector) * self.norm_action
        self.action_norm_vector = np.asarray(action_norm_vector)
    
    def _seed(self, seed=None):
      self.np_random, seed = seeding.np_random(seed)
      return [seed]
   
    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs