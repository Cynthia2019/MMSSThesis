from agent import DRLAgent
from helpers import *
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import EvalCallback
import gym
import optuna
from load_data import load_data


class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need.
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


def train_agent(trial, wrapped_env, model_name, model_params, break_step):

    agent = DRLAgent(wrapped_env)
    model_kwargs = {
        "n_steps": model_params["n_steps"],
        "ent_coef": model_params["ent_coef"],
        "learning_rate": model_params["learning_rate"],
        "batch_size": model_params["batch_size"],
    }
    policy_kwargs = {
        "net_arch": [model_params["net_arch"], model_params["net_arch"]],
    }
    model = agent.get_model(model_name, model_kwargs=model_kwargs, policy_kwargs=policy_kwargs)
    print("Training...")
    eval_callback = TrialEvalCallback(
        eval_env=wrapped_env,
        trial=trial,
    )
    model = agent.train_model(model, f"{model_name}_train", total_timesteps=break_step, callback=eval_callback)

    return agent, model

def test_agent(price_array_test, wrapped_env, model, agent, model_name): 
    agent.DRL_prediction(model, wrapped_env)
    account_value = pd.read_csv(f'results/account_value_test_{model_name}_.csv')['account_value']
    lookback = 1
    indice_start = lookback - 1
    indice_end = len(price_array_test) - lookback

    data_points_per_year = 12 * 24 * 365
    account_value_eqw, eqw_rets_tmp, eqw_cumrets = compute_eqw(price_array_test, indice_start, indice_end)
    dataset_size = np.shape(eqw_rets_tmp)[0]
    factor = data_points_per_year / dataset_size
    sharpe_eqw, _ = sharpe_iid(eqw_rets_tmp, bench=0, factor=factor, log=False)

    account_value = np.array(account_value)
    drl_rets_tmp = account_value[1:] - account_value[:-1]
    sharpe_bot, _ = sharpe_iid(drl_rets_tmp, bench=0, factor=factor, log=False)

    return sharpe_bot, sharpe_eqw, drl_rets_tmp