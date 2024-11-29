# File: Rl/environments/EnvironmentFinance.py

import numpy as np

class FinanceEnvironment:
    def __init__(self, ticker='AAPL', start_date='2020-01-01', end_date='2020-12-31', initial_cash=10000, max_shares=100, data=None):
        self.initial_cash = initial_cash
        self.max_shares = max_shares
        if data is not None:
            self.prices = data
            print("Loaded prices from provided data.")
        else:
            self.prices = self._generate_synthetic_prices(ticker, start_date, end_date)
            print("Generated synthetic prices.")
        if not hasattr(self, 'prices') or self.prices is None or len(self.prices) == 0:
            raise ValueError("Failed to initialize prices.")
        self.total_steps = len(self.prices)
        self.reset()

    def _generate_synthetic_prices(self, ticker, start_date, end_date):
        np.random.seed(42)
        time_interval = 200
        base_price = 300
        prices = base_price + np.cumsum(np.random.normal(0, 1, time_interval))
        print(f"Generated synthetic prices: {prices[:5]} ... {prices[-5:]}")
        return prices

    def reset(self):
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares = 0
        self.portfolio_value = self.cash
        self.portfolio_history = [self.portfolio_value]
        print("Environment reset.")
        return self._get_state()

    def step(self, action):
        done = False
        reward = 0
        current_price = self.prices[self.current_step]
        print(f"Step {self.current_step}: Price={current_price:.2f}, Action={action:.2f}")
        if action < 0:
            sell_qty = min(self.shares, abs(action))
            self.shares -= sell_qty
            self.cash += sell_qty * current_price
            reward = sell_qty * current_price
            print(f"Sold {sell_qty} shares at {current_price:.2f}, Cash={self.cash:.2f}, Shares={self.shares}")
        elif action > 0:
            affordable_qty = min(int(self.cash / current_price), self.max_shares - self.shares)
            buy_qty = min(abs(action), affordable_qty)
            self.shares += buy_qty
            self.cash -= buy_qty * current_price
            reward = -buy_qty * current_price
            print(f"Bought {buy_qty} shares at {current_price:.2f}, Cash={self.cash:.2f}, Shares={self.shares}")
        self.current_step += 1
        if self.current_step >= len(self.prices) - 1:
            done = True
        self.portfolio_value = self.cash + self.shares * self.prices[self.current_step]
        self.portfolio_history.append(self.portfolio_value)
        print(f"Portfolio value updated to {self.portfolio_value:.2f}")
        if done:
            reward = self.portfolio_value - self.initial_cash
            print(f"Final step reached. Reward: {reward:.2f}")
        return self._get_state(), reward, done

    def _get_state(self):
        return np.array([self.prices[self.current_step], self.cash, self.shares])

    def render(self):
        print(f"Step: {self.current_step}")
        print(f"Price: {self.prices[self.current_step]:.2f}")
        print(f"Cash: {self.cash:.2f}")
        print(f"Shares: {self.shares}")
        print(f"Portfolio Value: {self.portfolio_value:.2f}")
        print("-" * 30)
