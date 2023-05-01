import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from yahooDownloader import YahooDownloader
import yfinance as yf
from copy import deepcopy
from load_data import load_data, preprocess_data, df_to_array
from helpers import *
import pickle


def get_baseline(ticker, start, end):
    return YahooDownloader(
        start_date=start, end_date=end, ticker_list=[ticker]
    ).fetch_data()

def get_daily_return(df, value_col_name="account_value"):
    df = deepcopy(df)
    df["daily_return"] = df[value_col_name].pct_change(1)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True, drop=True)
    df.index = df.index.tz_localize("UTC")
    return pd.Series(df["daily_return"], index=df.index)

def download_CVIX(trade_start_date, trade_end_date):
    trade_start_date = trade_start_date[:10]
    trade_end_date = trade_end_date[:10]
    TIME_INTERVAL = '60m'
    CVOL_df = YahooDownloader(start_date=trade_start_date, end_date=trade_end_date, ticker_list=['CVOL-USD']).fetch_data()
    # YahooProcessor = Yahoofinance('yahoofinance', trade_start_date, trade_end_date, TIME_INTERVAL)
    # CVOL_df = YahooProcessor.download_data(['CVOL-USD'])
    CVOL_df.set_index('date', inplace=True)
    CVOL_df.index = pd.to_datetime(CVOL_df.index)
    CVOL_df = CVOL_df.resample('5Min').interpolate(method='linear')
    return CVOL_df['close']

def backtest_plot(): 
    # load data
    # df = load_data('processed/trade_data')
    with open('processed/trade_data', 'rb') as handle: 
        df = pickle.load(handle)
    price_array, tech_array, time_array = df_to_array(df)
    # load backtest data from csv file
    a2c_account_value = pd.read_csv("results/account_value_trade_a2c_.csv")
    ppo_account_value = pd.read_csv("results/account_value_trade_ppo_.csv")
    ppo_recurrent_account_value = pd.read_csv("results/account_value_trade_recurrent_.csv")
    drl_cumrets_list = []
    model_names_list = ['A2C', 'PPO', "RecurrentPPO"]

    #     unique_ticker = df.tic.unique()
    # if_first_time = True
    # ticker_list = unique_ticker
    # for tic in unique_ticker:
    #     if if_first_time:
    #         price_array = df[df.tic == tic][['close']].values
    #         tech_array = df[df.tic == tic][tech_indicator_list].values
    #         if_first_time = False
    #     else:
    #         price_array = np.hstack([price_array, df[df.tic == tic][['close']].values])
    #         tech_array = np.hstack([tech_array, df[df.tic == tic][tech_indicator_list].values])
    #     time_array = df[df.tic == ticker_list[0]].index

    unique_ticker = df["tic"].unique()
    if_first_time = True
    print(df.shape)
    for tic in unique_ticker:
        if if_first_time: 
            turbulence_array = df[df["tic"] == tic][['turbulence']].values
            if_first_time = False
        else:
            turbulence_array = np.hstack([turbulence_array, df[df["tic"] == tic][['turbulence']].values])

    start_date = a2c_account_value.loc[0, "date"][:10]
    end_date = a2c_account_value.loc[a2c_account_value.shape[0] - 1, "date"][:10]

    # compute equal-weight portfolio return
    # Compute Sharpe's of each coin
    indice_start = 0
    indice_end = len(price_array)
    account_value_eqw, ewq_rets, eqw_cumrets = compute_eqw(price_array, indice_start, indice_end)

    # Compute annualization factor
    data_points_per_year = 12 * 24 * 365
    dataset_size = np.shape(ewq_rets)[0]
    factor = data_points_per_year / dataset_size
    
    # Write buy-and-hold strategy
    eqw_annual_ret, eqw_annual_vol, eqw_sharpe_rat, eqw_vol = aggregate_performance_array(np.array(ewq_rets),
                                                                                                factor)
    write_metrics_to_results('Buy-and-Hold',
                                'plots_and_metrics/test_metrics.txt',
                                eqw_cumrets,
                                eqw_annual_ret,
                                eqw_annual_vol,
                                eqw_sharpe_rat,
                                eqw_vol,
                                'a'
                                )
    

    dfs = {
        'A2C': a2c_account_value,
        'PPO': ppo_account_value,
        'RecurrentPPO': ppo_recurrent_account_value
    }
    for model_name, df in dfs.items():
        # Compute DRL rets
        account_value_erl = np.array(df['account_value'])
        drl_rets = account_value_erl[1:] - account_value_erl[:-1]
        drl_cumrets = [x / account_value_erl[0] - 1 for x in account_value_erl]
        drl_cumrets_list.append(drl_cumrets)
            # Then compute the actual metrics from the DRL agents
        drl_annual_ret, drl_annual_vol, drl_sharpe_rat, drl_vol = aggregate_performance_array(np.array(drl_rets), factor)
        write_metrics_to_results(model_name,
                                'plots_and_metrics/test_metrics.txt',
                                drl_cumrets,
                                drl_annual_ret,
                                drl_annual_vol,
                                drl_sharpe_rat,
                                drl_vol,
                                'a'
                                )

    drl_rets_array = np.transpose(np.vstack(drl_cumrets_list))
    # General 1
    plt.rcParams.update({'font.size': 22})
    plt.figure(dpi=300)
    f, ax1 = plt.subplots(figsize=(20, 8))
    line_width = 2
    ax1.plot(time_array[1:], eqw_cumrets[1:], linewidth=line_width, label='Equal-weight', color='blue')

    CVIX_df = download_CVIX(start_date, end_date)
    CVIX_df = CVIX_df.reset_index()
    CVIX_df = pd.merge(time_array, CVIX_df, left_index=True, right_index=True, how='left')
    cvix_array = CVIX_df['close'].values
    # print(cvix_array)


    for i in range(np.shape(drl_rets_array)[1]):
        ax1.plot(time_array[1:], drl_rets_array[:, i], label=model_names_list[i], linewidth=line_width)
    ax1.legend(frameon=False, ncol=len(model_names_list) + 2, loc='upper left', bbox_to_anchor=(0, 1.11))
    ax1.patch.set_edgecolor('black')
    ax1.patch.set_linewidth(3)
    ax1.grid()
    # plt.legend()

    # # # Plot CVIX
    ax2 = ax1.twinx()
    ax2.plot(time_array[1:], cvix_array[1:], label='CVIX', color='black', linestyle='dashed', alpha=0.4)
    # ax2.legend(frameon=False, loc='upper right', bbox_to_anchor=(0.7, 1.17))
    ax2.patch.set_edgecolor('black')
    ax2.patch.set_linewidth(3)
    ax2.set_ylabel('CVIX')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=10))
    ax1.set_ylabel('Cumulative return')
    plt.xlabel('Date')
    plt.show()


if __name__ == "__main__":
    backtest_plot()
    