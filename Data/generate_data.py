#!/FinRL/Data/generate_data.py

import warnings
warnings.filterwarnings("ignore")

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
from Rl.preprocessors.yahoodownloader import YahooDownloader
from Rl.preprocessors.preprocessors import GroupByScaler

SAVE_PATH = "Data/parquet/"

TOP_BRL = [
    "VALE3.SA", "PETR4.SA", "ITUB4.SA", "BBDC4.SA",
    "BBAS3.SA", "RENT3.SA", "LREN3.SA", "PRIO3.SA",
    "WEGE3.SA", "ABEV3.SA"
]

portfolio_raw_df = YahooDownloader(start_date = '2011-01-01',
                                end_date = '2022-12-31',
                                ticker_list = TOP_BRL).fetch_data()
portfolio_norm_df = GroupByScaler(by="tic", scaler=MaxAbsScaler).fit_transform(portfolio_raw_df)
df_portfolio = portfolio_norm_df[["date", "tic", "close", "high", "low"]]

df_portfolio_train = df_portfolio[(df_portfolio["date"] >= "2011-01-01") & (df_portfolio["date"] < "2020-12-31")]
df_portfolio_train = pd.DataFrame(df_portfolio_train)
df_portfolio_train.to_parquet(SAVE_PATH + "df_portfolio_train.parquet")
df_portfolio_2016 = df_portfolio[(df_portfolio["date"] >= "2011-01-01") & (df_portfolio["date"] < "2016-12-31")]
df_portfolio_2017 = df_portfolio[(df_portfolio["date"] >= "2017-01-01") & (df_portfolio["date"] < "2017-12-31")]
df_portfolio_2018 = df_portfolio[(df_portfolio["date"] >= "2018-01-01") & (df_portfolio["date"] < "2018-12-31")]
df_portfolio_2019 = df_portfolio[(df_portfolio["date"] >= "2019-01-01") & (df_portfolio["date"] < "2019-12-31")]
df_portfolio_2020 = df_portfolio[(df_portfolio["date"] >= "2020-01-01") & (df_portfolio["date"] < "2020-12-31")]
df_portfolio_2021 = df_portfolio[(df_portfolio["date"] >= "2021-01-01") & (df_portfolio["date"] < "2021-12-31")]
df_portfolio_2022 = df_portfolio[(df_portfolio["date"] >= "2022-01-01") & (df_portfolio["date"] < "2022-12-31")]

df_portfolio_2020 = pd.DataFrame(df_portfolio_2020)
df_portfolio_2020.to_parquet(SAVE_PATH + "df_portfolio_2020.parquet")

df_portfolio_2021 = pd.DataFrame(df_portfolio_2021)
df_portfolio_2021.to_parquet(SAVE_PATH + "df_portfolio_2021.parquet")

df_portfolio_2022 = pd.DataFrame(df_portfolio_2022)
df_portfolio_2022.to_parquet(SAVE_PATH + "df_portfolio_2022.parquet")