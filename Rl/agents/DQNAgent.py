import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Define available activation functions
ACTIVATION_FUNCTIONS = {
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "tanh": nn.Tanh(),
    "leaky_relu": nn.LeakyReLU()
}

class DQNAgent:
    def __init__(self, state_size, action_size=1, architecture="simple", activation="relu", max_action=10):

        self.state_size = state_size
        self.action_size = action_size  
        self.memory = deque(maxlen=10000)  
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64 
        self.architecture = architecture
        self.activation = activation
        self.max_action = max_action 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()


        self.loss_history = []

    def _build_model(self):
 
        activation_fn = ACTIVATION_FUNCTIONS.get(self.activation, nn.ReLU())
        if self.architecture == "simple":
            return nn.Sequential(
                nn.Linear(self.state_size, 128),
                activation_fn,
                nn.Linear(128, self.action_size)
            )
        elif self.architecture == "medium":
            return nn.Sequential(
                nn.Linear(self.state_size, 256),
                activation_fn,
                nn.Linear(256, 128),
                activation_fn,
                nn.Linear(128, self.action_size)
            )
        elif self.architecture == "complex":
            return nn.Sequential(
                nn.Linear(self.state_size, 512),
                activation_fn,
                nn.Linear(512, 256),
                activation_fn,
                nn.Linear(256, 128),
                activation_fn,
                nn.Linear(128, self.action_size)
            )
        else:
            raise ValueError("Unknown architecture type. Choose 'simple', 'medium', or 'complex'.")

    def remember(self, state, action, reward, next_state, done):

        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        
        if random.random() <= self.epsilon:
            
            return random.uniform(-self.max_action, self.max_action)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.model(state).item()
            
            action = max(-self.max_action, min(self.max_action, action))
        return action

    def replay(self):

        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        batch_loss = 0.0

        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            action = torch.FloatTensor([action]).to(self.device)
            reward = torch.FloatTensor([reward]).to(self.device)
            done = torch.FloatTensor([done]).to(self.device)

            target = reward
            if not done:
                target = reward + self.gamma * self.model(next_state).detach()

            current = self.model(state).squeeze()

            loss = self.criterion(current, target)
            batch_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        avg_loss = batch_loss / self.batch_size
        self.loss_history.append(avg_loss)

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
                    print(f"Episode: {e + 1}/{episodes}, Total Reward: {total_reward:.2f}, Final Portfolio: ${env.portfolio_value:.2f}, Epsilon: {self.epsilon:.2f}")
                    break
            self.replay()

    def plot_loss(self, filename='loss_animation.gif'):

        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history, label='Loss')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.legend()
        plt.savefig('loss_plot.png') 
        plt.close()

        image = Image.open('loss_plot.png')
        image.save(filename, save_all=True, append_images=[image], duration=500, loop=0)
        print(f"Loss animation saved as {filename}")

    def save_learning_animation(self, states, actions, portfolio_values, filename='learning_animation.gif'):

        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(0, len(portfolio_values))
        min_val = min(portfolio_values) - 100
        max_val = max(portfolio_values) + 100
        ax.set_ylim(min_val, max_val)
        ax.set_title("Agent's Portfolio Value Over Time")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Portfolio Value (USD)")

        line, = ax.plot([], [], lw=2, label='Portfolio Value')
        action_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')
        portfolio_text = ax.text(0.05, 0.90, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')

        def update(frame):
            line.set_data(range(frame), portfolio_values[:frame])
            if frame - 1 < len(actions):
                action = actions[frame - 1]
                action_str = f"Action: {'Buy' if action > 0 else 'Sell' if action < 0 else 'Hold'} {abs(action):.2f} shares"
                action_text.set_text(action_str)
                portfolio_text.set_text(f"Portfolio Value: ${portfolio_values[frame - 1]:.2f}")
            return line, action_text, portfolio_text

        ani = FuncAnimation(fig, update, frames=len(portfolio_values), blit=True, interval=100)
        ani.save(filename, writer=PillowWriter(fps=10))
        plt.close()
        print(f"Learning animation saved as {filename}")

    def save_model(self, filename='dqn_model.pth'):

        torch.save(self.model.state_dict(), filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename='dqn_model.pth'):
        self.model.load_state_dict(torch.load(filename, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {filename}")

    def evaluate_agent(self, env, render=False):

        state = env.reset()
        test_portfolios = []
        actions_taken = []
        states = []
        for step in range(env.total_steps - 1):
            action = self.choose_action(state)
            next_state, reward, done = env.step(action)
            if render:
                states.append(state)
                actions_taken.append(action)
                test_portfolios.append(env.portfolio_value)
            state = next_state
            if done:
                break
        print(f"Final Portfolio Value: ${env.portfolio_value:.2f}")
        print(f"Cumulative Profit/Loss: ${env.portfolio_value - env.initial_cash:.2f}")
        return test_portfolios, states, actions_taken
    
    def compute_metrics(self, portfolio_values):

        portfolio = np.array(portfolio_values)
        initial_value = portfolio[0]
        final_value = portfolio[-1]
        
        cumulative_return = (final_value / initial_value) - 1
        returns = np.diff(portfolio) / portfolio[:-1]
        
        if returns.std() != 0:
            sharpe_ratio = returns.mean() / returns.std()
        else:
            sharpe_ratio = 0.0
        
        cumulative_max = np.maximum.accumulate(portfolio)
        drawdowns = (cumulative_max - portfolio) / cumulative_max
        max_drawdown = np.max(drawdowns)
        
        if len(returns) > 0:
            performance_stability = np.sum(returns > 0) / len(returns)
        else:
            performance_stability = 0.0
        
        metrics = {
            'Cumulative Return': cumulative_return,
            'Maximum Drawdown': max_drawdown,
            'Sharpe Ratio': sharpe_ratio,
            'Performance Stability': performance_stability
        }
        
        return metrics
