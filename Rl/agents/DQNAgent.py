import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size, architecture="simple"):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.architecture = architecture

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def _build_model(self):
        if self.architecture == "simple":
            return nn.Sequential(
                nn.Linear(self.state_size, 24),
                nn.ReLU(),
                nn.Linear(24, self.action_size)
            )
        elif self.architecture == "medium":
            return nn.Sequential(
                nn.Linear(self.state_size, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, self.action_size)
            )
        elif self.architecture == "complex":
            return nn.Sequential(
                nn.Linear(self.state_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, self.action_size)
            )
        else:
            raise ValueError("Unknown architecture type. Choose 'simple', 'medium', or 'complex'.")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            reward = torch.FloatTensor([reward]).to(self.device)
            done = torch.FloatTensor([done]).to(self.device)

            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state).detach())

            q_values = self.model(state)
            target_f = q_values.clone().detach()
            target_f[action] = target

            loss = self.criterion(q_values, target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train_agent(self, env, episodes=50):
        for e in range(episodes):
            state = env.reset()
            total_reward = 0
            for time in range(env.total_steps - 1):
                action = self.choose_action(state)
                next_state, reward, done = env.step(action)
                self.remember(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state
                if done:
                #    print(f"Episode: {e + 1}/{episodes}, Total Reward: {total_reward}, Final Portfolio: {env.portfolio_value}")
                    break
            self.replay()
