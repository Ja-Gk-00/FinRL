import numpy as np
import pandas as pd
import random

class StockTradingEnv:
    def __init__(self, data, initial_balance=10000):
        self.data = data
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.total_shares_bought = 0
        self.total_shares_sold = 0
        self.total_profit = 0
        self.action_space = [0, 1, 2]  # 0 = hold, 1 = buy, 2 = sell

    def reset(self):
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_shares_bought = 0
        self.total_shares_sold = 0
        self.total_profit = 0
        self.current_step = 0
        return self._next_observation()

    def _next_observation(self):
        obs = self.data.iloc[self.current_step]
        return obs.values

    def step(self, action):
        current_price = self.data['close'].iloc[self.current_step]
        
        if action == 1:  # Buy
            shares_bought = self.balance // current_price
            self.balance -= shares_bought * current_price
            self.shares_held += shares_bought
            self.total_shares_bought += shares_bought
        
        elif action == 2:  # Sell
            self.balance += self.shares_held * current_price
            self.total_shares_sold += self.shares_held
            self.shares_held = 0

        reward = self._calculate_reward(current_price)
        
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        next_state = self._next_observation()

        return next_state, reward, done

    def _calculate_reward(self, current_price):
        total_value = self.balance + (self.shares_held * current_price)
        reward = total_value - self.initial_balance
        return reward