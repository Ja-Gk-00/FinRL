# File: Rl/environments/MultiStockContinuousEnvironment.py

import gym
from gym import spaces
import numpy as np
import yfinance as yf

class MultiStockContinuousEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, tickers=['AAPL', 'GOOGL'], start_date='2020-01-01', end_date='2020-12-31',
                 initial_cash=100000, max_shares=1000, max_buy=100, max_sell=100):

        super(MultiStockContinuousEnv, self).__init__()
        
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.max_shares = max_shares
        self.max_buy = max_buy
        self.max_sell = max_sell
        
        self._fetch_data()
        self.action_space = spaces.Box(low=-self.max_sell, high=self.max_buy, shape=(len(self.tickers),), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(len(self.tickers)*2 +1,), dtype=np.float32)
        
        self.reset()
    
    def _fetch_data(self):
        self.prices = {}
        for ticker in self.tickers:
            try:
                data = yf.download(ticker, start=self.start_date, end=self.end_date)
                if data.empty:
                    raise ValueError(f"No data fetched for {ticker}.")
                self.prices[ticker] = data['Close'].values
                print(f"Fetched {len(self.prices[ticker])} price points for {ticker}.")
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
                raise e
        
        lengths = [len(prices) for prices in self.prices.values()]
        if len(set(lengths)) != 1:
            raise ValueError("All stocks must have the same number of time steps.")
        self.total_steps = lengths[0]
    
    def reset(self):

        self.current_step = 0
        self.cash = self.initial_cash
        self.shares = {ticker: 0 for ticker in self.tickers}
        self.portfolio_value = self.cash
        return self._get_observation()
    
    def step(self, action):
        done = False
        reward = 0.0 
        
        for idx, ticker in enumerate(self.tickers):
            act = action[idx]
            current_price = float(self.prices[ticker][self.current_step]) 
            
            if act > 0: 
                buy_qty = min(int(act), int(self.cash / current_price), self.max_shares - self.shares[ticker])
                if buy_qty > 0:
                    self.shares[ticker] += buy_qty
                    self.cash -= buy_qty * current_price
                    reward -= buy_qty * current_price 
                    print(f"Bought {buy_qty} shares of {ticker} at {current_price:.2f}")
                else:
                    print(f"Attempted to buy {act} shares of {ticker}, but insufficient cash or max shares reached.")
            elif act < 0: 
                sell_qty = min(int(-act), self.shares[ticker], self.max_shares)
                if sell_qty > 0:
                    self.shares[ticker] -= sell_qty
                    self.cash += sell_qty * current_price
                    reward += sell_qty * current_price  
                    print(f"Sold {sell_qty} shares of {ticker} at {current_price:.2f}")
                else:
                    print(f"Attempted to sell {-act} shares of {ticker}, but no shares held.")
        
        self.current_step += 1
        if self.current_step >= self.total_steps - 1:
            done = True
        
        portfolio_value = self.cash
        for ticker in self.tickers:
            portfolio_value += self.shares[ticker] * self.prices[ticker][self.current_step]
        self.portfolio_value = float(portfolio_value) 
        self.portfolio_values = self.portfolio_values + [self.portfolio_value] if hasattr(self, 'portfolio_values') else [self.portfolio_value]
    
        if done:
            reward += float(self.portfolio_value - self.initial_cash)
        
        assert isinstance(reward, (float, int)), f"Reward is not a scalar float: {reward}"
        
        observation = self._get_observation()
        
        info = {'portfolio_value': self.portfolio_value}
        
        return observation, reward, done, info
    
    def _get_observation(self):

        obs = []
        for ticker in self.tickers:
            obs.append(self.prices[ticker][self.current_step])
            obs.append(self.shares[ticker])
        obs.append(self.cash)
        return np.array(obs, dtype=np.float32)
    
    def render(self, mode='human'):
        print(f"Step: {self.current_step}")
        print(f"Cash: {self.cash:.2f}")
        for ticker in self.tickers:
            current_price = self.prices[ticker][self.current_step]
            shares = self.shares[ticker]
            print(f"{ticker} - Price: {current_price:.2f}, Shares: {shares}")
        print(f"Portfolio Value: {self.portfolio_value:.2f}")
        print("-" * 50)
