from Rl.agents.PortfolioOptimizationAgents.models import DRLAgent
from Rl.agents.PortfolioOptimizationAgents.architectures import EIIE
from Rl.environments.EnvironmentPortfolioOptimization import PortfolioOptimizationEnv
from Rl.preprocessors.yahoodownloader import YahooDownloader
from Rl.preprocessors.preprocessors import GroupByScaler
from sklearn.preprocessing import MaxAbsScaler

import numpy as np
import pandas as pd
import torch

model_kwargs = {
    "lr": 0.01,
    "policy": EIIE,
}

policy_kwargs = {
    "k_size": 3,
    "time_window": 50,
}

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

df_portfolio_train = pd.read_parquet("Data/parquet/df_portfolio_train.parquet")

train_env = PortfolioOptimizationEnv(
        df_portfolio_train,
        initial_amount=100000,
        comission_fee_pct=0.0025,
        time_window=50,
        features=["close", "high", "low"],
        normalize_df=None
    )

model = DRLAgent(train_env).get_model("pg", device, model_kwargs, policy_kwargs)

DRLAgent.train_model(model, episodes=40)
print(f"Model successfuly trained.")

torch.save(model.train_policy.state_dict(), "models_saved/pytorch_format/train_saved_EIIE.pt")
print(f"Model saved into models_saved/pytorch_format")