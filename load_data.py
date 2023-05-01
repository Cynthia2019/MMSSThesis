import pickle
from feature_engineer import BinanceFeatureEngineer
import numpy as np
def load_data(name):
    with open(name, 'rb') as handle: 
        df = pickle.load(handle)
        df = df[~df.tic.isin(['AVAXUSDT', 'LINKUSDT', 'MATICUSDT', 'NEARUSDT'])]
        df.index = df.index.rename('date')
        df = df.reset_index()
    return df

def dump_data(df, name): 
    with open(name, 'wb') as handle: 
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

def preprocess_data(df): 
    fe = BinanceFeatureEngineer(use_technical_indicator=False, 
                                use_turbulence=True, 
                                user_defined_feature=False, 
                                use_vix=False)
    df = fe.preprocess_data(df)
    df = df.copy()
    df = df.fillna(0)
    df = df.replace(np.inf, 0)
    return df

def df_to_array(df, if_vix=False):
    tech_indicator_list = list(df.columns)
    tech_indicator_list.remove('tic')
    if 'date' in df.columns:
        tech_indicator_list.remove('date')
    tech_indicator_list.remove('close')
    if 'turbulence' in df.columns: 
        tech_indicator_list.remove('turbulence')

    if 'CVIX' in df.columns: 
        tech_indicator_list.remove('CVIX')
    print('adding technical indiciators (no:', len(tech_indicator_list), ') :', tech_indicator_list)

    unique_ticker = df.tic.unique()
    if_first_time = True
    ticker_list = unique_ticker
    for tic in unique_ticker:
        if if_first_time:
            price_array = df[df.tic == tic][['close']].values
            tech_array = df[df.tic == tic][tech_indicator_list].values
            if_first_time = False
        else:
            price_array = np.hstack([price_array, df[df.tic == tic][['close']].values])
            tech_array = np.hstack([tech_array, df[df.tic == tic][tech_indicator_list].values])
        time_array = df[df.tic == ticker_list[0]].date

    assert price_array.shape[0] == tech_array.shape[0]

    return price_array, tech_array, time_array
