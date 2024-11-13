import numpy as np
import random

class AgentSimple:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice([0, 1, 2]) 
        return np.argmax(self.q_table[state]) 

    def learn(self, state, action, reward, next_state):
        q_update = reward + self.discount_factor * np.max(self.q_table[next_state])
        self.q_table[state][action] = (1 - self.learning_rate) * self.q_table[state][action] + self.learning_rate * q_update

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def train_agent(self, env, episodes=100):
        for e in range(episodes):
            state = env.reset()
            total_reward = 0
            for time in range(len(env.data) - 1):
                action = self.choose_action(time)
                next_state, reward, done = env.step(action)
                self.learn(time, action, reward, time + 1)
                total_reward += reward

                if done:
                    print(f"Episode: {e + 1}/{episodes}, Total Reward: {total_reward}")
                    break
