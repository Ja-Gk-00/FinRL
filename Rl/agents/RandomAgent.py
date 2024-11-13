import numpy as np
import random
import csv

class RandomAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.action_log = []
        self.reward_log = []
        self.step_log = []
        self.portfolio_log = []

    def choose_action(self, state):
        return random.choice(range(self.action_size))

    def learn(self, state, action, reward, next_state):
        pass

    def train_agent(self, env, episodes=100):
        for e in range(episodes):
            state = env.reset()
            total_reward = 0
            for time in range(len(env.data) - 1):
                action = self.choose_action(state)
                next_state, reward, done = env.step(action)
                self.learn(state, action, reward, next_state)
                self.action_log.append(action)
                self.reward_log.append(reward)
                self.step_log.append(time)
                self.portfolio_log.append(env.portfolio_value)
                total_reward += reward
                state = next_state
                if done:
                    print(f"RandomAgent Episode: {e + 1}/{episodes}, Total Reward: {total_reward}, Final Portfolio: {env.portfolio_value}")
                    break

    def save_logs(self, filename='random_agent_logs.csv'):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Step', 'Action', 'Reward', 'Portfolio Value'])
            for step, action, reward, portfolio in zip(self.step_log, self.action_log, self.reward_log, self.portfolio_log):
                writer.writerow([step, action, reward, portfolio])
