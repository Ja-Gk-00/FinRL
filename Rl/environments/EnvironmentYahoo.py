import yfinance as yf
import random
import pandas as pd

class FinanceEnv:
    def __init__(self, ticker, start_date, end_date, initial_cash=10000):
        self.ticker = ticker
        self.data = yf.download(ticker, start=start_date, end=end_date)
        if 'Close' not in self.data.columns:
            raise ValueError("Downloaded data does not contain 'Close' prices.")
        self.current_step = 0
        self.total_steps = len(self.data)
        self.done = False
        self.initial_cash = initial_cash
        self.cash = float(initial_cash)
        self.shares = 0
        self.portfolio_value = float(initial_cash)

    def reset(self):
        self.current_step = 0
        self.done = False
        self.cash = float(self.initial_cash)
        self.shares = 0
        self.portfolio_value = float(self.initial_cash)
        return self.current_step

    def step(self, action):
        if self.current_step >= self.total_steps - 1:
            self.done = True
            return self.current_step, 0, self.done

        current_price = self.data['Close'].iloc[self.current_step]
        if isinstance(current_price, pd.Series):
            current_price = float(current_price[self.ticker])

        current_price = float(current_price)

        reward = 0

        if action == 0:  # Buy
            if self.cash >= current_price:
                shares_to_buy = self.cash // current_price
                self.shares += int(shares_to_buy)
                self.cash -= shares_to_buy * current_price
        elif action == 1:  # Sell
            if self.shares > 0:
                self.cash += self.shares * current_price
                self.shares = 0
        #Hold; do nothing

        self.current_step += 1

        next_price = self.data['Close'].iloc[self.current_step]
        if isinstance(next_price, pd.Series):
            next_price = float(next_price[self.ticker])

        next_price = float(next_price)

        new_portfolio_value = self.cash + self.shares * next_price
        reward = new_portfolio_value - self.portfolio_value
        self.portfolio_value = new_portfolio_value

        if self.current_step >= self.total_steps - 1:
            self.done = True

        return self.current_step, reward, self.done
