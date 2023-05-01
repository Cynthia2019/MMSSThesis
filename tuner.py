import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
import operator as op
import joblib
from functools import reduce
from CryptoEnv import CryptoEnv
from functionCPCV import *
from config_main import *
from load_data import load_data, preprocess_data, df_to_array
from feature_engineer import data_split
from train_test import train_agent, test_agent
from agent import DRLAgent
from stable_baselines3.common.vec_env import DummyVecEnv
import os 
from distutils.dir_util import copy_tree
import pickle

import warnings
warnings.filterwarnings('ignore')

def nCr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2


def sample_hyperparams(trial):
    sampled_model_params = {
        "learning_rate": trial.suggest_categorical("learning_rate", [3e-2, 2.3e-2, 1.5e-2, 7.5e-3, 5e-6]),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512, 1280]),
        "gamma": trial.suggest_categorical("gamma", [0.85, 0.95, 0.99, 0.999]),
        "net_arch": trial.suggest_categorical("net_arch", [2 ** 9, 2 ** 10, 2 ** 11, 2 ** 12]),
        "target_step": trial.suggest_categorical("target_step",
                                                 [5e4]),
        "n_steps": trial.suggest_categorical("n_steps", [2, 5, 10, 15, 20]),
        "ent_coef": trial.suggest_categorical("ent_coef", [0.005, 0.01, 0.05, 0.1]),
        "eval_time_gap": trial.suggest_categorical("eval_time_gap", [60]),
        "break_step": trial.suggest_categorical("break_step", [100])
    }

    # environment normalization and lookback
    sampled_env_params = {
        "lookback": trial.suggest_categorical("lookback", [1]),
        "norm_cash": trial.suggest_categorical("norm_cash", [2 ** -12]),
        "norm_stocks": trial.suggest_categorical("norm_stocks", [2 ** -8]),
        "norm_tech": trial.suggest_categorical("norm_tech", [2 ** -15]),
        "norm_reward": trial.suggest_categorical("norm_reward", [2 ** -10]),
        "norm_action": trial.suggest_categorical("norm_action", [10000])
    }
    return sampled_model_params, sampled_env_params

def set_Pandas_Timedelta(TIMEFRAME):
    timeframe_to_delta = {'1m': pd.Timedelta(minutes=1),
                          '5m': pd.Timedelta(minutes=5),
                          '10m': pd.Timedelta(minutes=10),
                          '30m': pd.Timedelta(minutes=30),
                          '1h': pd.Timedelta(hours=1),
                          '1d': pd.Timedelta(days=1),
                          }
    if TIMEFRAME in timeframe_to_delta:
        return timeframe_to_delta[TIMEFRAME]
    else:
        raise ValueError('Timeframe not supported yet, please manually add!')


def print_best_trial(study, trial):
    
    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))


def set_pickle_attributes(trial, model_name, TIMEFRAME, TRAIN_START_DATE, TRAIN_END_DATE, VAL_START_DATE, VAL_END_DATE, TICKER_LIST, TECHNICAL_INDICATORS_LIST, name_folder, name_test, study):
    # user attributes for saving in the pickle model file later
     # Display true values.
    trial.set_user_attr("model_name", model_name)
    trial.set_user_attr("timeframe", TIMEFRAME)
    trial.set_user_attr("train_start_date", TRAIN_START_DATE)
    trial.set_user_attr("train_end_date", TRAIN_END_DATE)
    trial.set_user_attr("test_start_date", VAL_START_DATE)
    trial.set_user_attr("test_end_date", VAL_END_DATE)
    trial.set_user_attr("ticker_list", TICKER_LIST)
    trial.set_user_attr("technical_indicator_list", TECHNICAL_INDICATORS_LIST)
    trial.set_user_attr("name_folder", name_folder)
    trial.set_user_attr("name_test", name_test)
    joblib.dump(study, f'train_results/{name_folder}/' + 'study.pkl')

def write_logs(name_folder, model_name, trial, cwd, erl_params, env_params, num_paths, n_total_groups, n_splits):
    path_logs = './train_results/' + name_folder + '/logs.txt'
    with open(path_logs, 'a') as f:
        f.write('\n' + 'MODEL NAME: ' + model_name + '\n')
        f.write('TRIAL NUMBER: ' + str(trial.number) + '\n')
        f.write('CWD: ' + cwd + '\n')
        f.write(str(erl_params) + '\n')
        f.write(str(env_params) + '\n')
        f.write('\n' + 'TIME START OUTER: ' + str(datetime.now()) + '\n')

        f.write('\n######### CPCV Settings #########' + '\n')
        f.write("Paths  : " + str(num_paths) + '\n')
        f.write("N      : " + str(n_total_groups) + '\n')
        f.write("splits : " + str(n_splits) + '\n\n')
    return path_logs

def setup_CPCV(df, erl_params, tech_array, time_array, NUM_PATHS, K_TEST_GROUPS, TIMEFRAME):
    env = CryptoEnv
    break_step = erl_params["break_step"]

    # Setup Purged CombinatorialCross-Validation
    num_paths = NUM_PATHS
    k_test_groups = K_TEST_GROUPS
    n_total_groups = num_paths + 1
    t_final = 10
    embargo_td = set_Pandas_Timedelta(TIMEFRAME) * t_final * 5
    n_splits = np.array(list(itt.combinations(np.arange(n_total_groups), k_test_groups))).reshape(-1, k_test_groups)
    n_splits = len(n_splits)

    cv = CombPurgedKFoldCV(n_splits=n_total_groups, n_test_splits=k_test_groups, embargo_td=embargo_td)

    # Set placeholder target variable
    data = pd.DataFrame(tech_array)
    data = data.set_index(time_array)
    data.drop(data.tail(t_final).index, inplace=True)
    y = pd.Series([0] * df.shape[0])
    y = y.reindex(data.index)
    y = y.squeeze()

    # prediction and evaluation times
    prediction_times = pd.Series(data.index, index=data.index)
    evaluation_times = pd.Series(data.index, index=data.index)

        # Compute paths
    is_test, paths, _ = back_test_paths_generator(data, y, cv, data.shape[0], n_total_groups, k_test_groups,
                                                  prediction_times, evaluation_times, verbose=False)

    return cv, env, data, y, num_paths, paths, n_total_groups, n_splits, break_step, prediction_times, evaluation_times


def objective(trial, name_test, model_name, cwd, res_timestamp):
    # Set full name_folder
    name_folder = res_timestamp + '_' + name_test

    train_data = load_data('data/train_data')
    train_data = preprocess_data(train_data)
    train_data = train_data.drop(columns=['CVIX'])
    TECHNICAL_INDICATORS_LIST = ['dx', 'ht_dcphase', 'obv', 'rsi', 'ultosc', 'volume']
    TRAIN_START_DATE = train_data.loc[0, 'date']
    TRAIN_END_DATE = train_data.loc[train_data.shape[0] - 1, 'date']

    train_data.set_index('date', inplace=True, drop=True)

    # train_data = data_split(train_data, TRAIN_START_DATE, TRAIN_END_DATE)
    # date as index
    train_price_array, train_tech_array, train_time_array = df_to_array(train_data)


    set_pickle_attributes(trial, model_name, TIMEFRAME, TRAIN_START_DATE, TRAIN_END_DATE, VAL_START_DATE, VAL_END_DATE,
                          TICKER_LIST, TECHNICAL_INDICATORS_LIST, name_folder, name_test, study)
    
    # Sample set of hyperparameters
    model_params, env_params = sample_hyperparams(trial)

    # Setup Combinatorial Purged Cross-Validation
    cpcv, \
        env, \
        data, y, \
        num_paths, \
        paths, \
        n_total_groups, \
        n_splits, \
        break_step, \
        prediction_times, \
        evaluation_times = setup_CPCV(train_data, model_params, train_tech_array, train_time_array, NUM_PATHS, K_TEST_GROUPS,
                                      TIMEFRAME)
    train_data = train_data.reset_index()
    train_data = data_split(train_data, TRAIN_START_DATE, TRAIN_END_DATE)
    print(train_data.shape, data.shape)
    
    # initiate logs for tracking behaviour during training
    path_logs = write_logs(name_folder, model_name, trial, cwd, model_params, env_params, num_paths, n_total_groups,
                           n_splits)
    
    # CPCV Split function eval
    #######################################################################################################
    #######################################################################################################

    # CV loop
    sharpe_list_bot = []
    sharpe_list_ewq = []
    drl_rets_val_list = []

    for split, (train_indicies, test_indicies) in enumerate(
        cpcv.split(data, y, pred_times=prediction_times, eval_times=evaluation_times)):
        
        with open(path_logs, 'a') as f:
            f.write('TIME START INNER: ' + str(datetime.now())) 


        # train model 
        train_env = env(
            config={
                "price_array": train_price_array[train_indicies, :],
                "tech_array": train_tech_array[train_indicies, :],
                "df": train_data,
            }, 
            env_params=env_params,
            tech_indicator_list=TECHNICAL_INDICATORS_LIST,
            mode="train",
            model_name=model_name
        )
        wrapped_train_env = DummyVecEnv([lambda: train_env])
        agent, model = train_agent(trial, wrapped_train_env, model_name, model_params, break_step)

        
        # test model
        test_env = env(
            config={
                "price_array": train_price_array[test_indicies, :],
                "tech_array": train_tech_array[test_indicies, :],
                "df": train_data,
            },
            env_params=env_params,
            tech_indicator_list=TECHNICAL_INDICATORS_LIST,
            mode="test",
            model_name=model_name
        )
        wrapped_test_env = DummyVecEnv([lambda: test_env])
        sharpe_bot, sharpe_eqw, drl_rets_tmp = test_agent(train_price_array[test_indicies, :], test_env, model, agent, model_name)
        
        sharpe_list_ewq.append(sharpe_eqw)
        sharpe_list_bot.append(sharpe_bot)

        with open(path_logs, 'a') as f:
            f.write('\n' + 'SPLIT: ' + str(split) + '     # Optimizing for Sharpe ratio!' + '\n')
            f.write('BOT:         ' + str(sharpe_bot) + '\n')
            f.write('HODL:        ' + str(sharpe_eqw) + '\n')
            f.write('TIME END INNER: ' + str(datetime.now()) + '\n\n')

        # Fill the backtesting prediction matrix
        drl_rets_val_list.append(drl_rets_tmp)
        trial.set_user_attr("price_array", train_price_array)
        trial.set_user_attr("tech_array", train_tech_array)
        trial.set_user_attr("time_array", train_time_array)

    # Hyperparameter bjective function eval
    #######################################################################################################
    #######################################################################################################

    # Matrices
    trial.set_user_attr("drl_rets_val_list", drl_rets_val_list)

    # Interesting values
    trial.set_user_attr("sharpe_list_bot", sharpe_list_bot)
    trial.set_user_attr("sharpe_list_ewq", sharpe_list_ewq)
    trial.set_user_attr("paths", paths)

    with open(path_logs, 'a') as f:
        f.write('\nHYPERPARAMETER EVAL || SHARPE AVG BOT    :  ' + str(np.mean(sharpe_list_bot)) + '\n')
        f.write('HYPERPARAMETER EVAL || SHARPE AVG HODL     : ' + str(np.mean(sharpe_list_ewq)) + '\n')
        f.write('DIFFERENCE                                 : ' + str(
            np.mean(sharpe_list_bot) - np.mean(sharpe_list_ewq)) + '\n')
        f.write('\n' + 'TIME END OUTER: ' + str(datetime.now()) + '\n')


    return np.mean(sharpe_list_bot) - np.mean(sharpe_list_ewq)


# Optuna
#######################################################################################################

def optimize(name_test, model_name):
    # Auto naming
    res_timestamp = 'res_' + str(datetime.now().strftime("%Y-%m-%d__%H_%M_%S"))
    name_test = f"{name_test}_CPCV_{model_name}_{TIMEFRAME}_{H_TRIALS}H_{round((no_candles_for_train + no_candles_for_val) / 1000)}k"
    cwd = f"./train_results/cwd_tests/{name_test}"
    path = f"./train_results/{res_timestamp}_{name_test}/"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(f"./train_results/{res_timestamp}_{name_test}/logs.txt", "w") as f:
        f.write(f"##################################  || {model_name} || ##################################")

    global study

    def obj_with_argument(trial):
        return objective(trial, name_test, model_name, cwd, res_timestamp)

    sampler = TPESampler(multivariate=True, seed=SEED_CFG)
    study = optuna.create_study(
        study_name=None,
        direction='maximize',
        sampler=sampler,
        pruner=HyperbandPruner(
            min_resource=1,
            max_resource=300,
            reduction_factor=3
        )
    )
    study.optimize(
        obj_with_argument,
        n_trials=H_TRIALS,
        catch=(ValueError,),
        callbacks=[print_best_trial]
    )


# Main
#######################################################################################################
name_model = 'ppo'
name_test = 'model'

print('\nStarting CPCV optimization with:')
print('drl algorithm:       ', name_model)
print('name_test:           ', name_test)

optimize(name_test, name_model)