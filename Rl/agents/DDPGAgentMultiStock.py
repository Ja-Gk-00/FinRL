import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.animation import FuncAnimation
from itertools import product

ACTIVATION_FUNCTIONS = {
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "tanh": nn.Tanh(),
    "leaky_relu": nn.LeakyReLU()
}

class Actor(nn.Module):
    def __init__(self, state_size, action_size, architecture="simple", activation="relu"):
        super(Actor, self).__init__()
        activation_fn = ACTIVATION_FUNCTIONS.get(activation, nn.ReLU())
        if architecture == "simple":
            self.network = nn.Sequential(
                nn.Linear(state_size, 128),
                activation_fn,
                nn.Linear(128, 128),
                activation_fn,
                nn.Linear(128, action_size),
                nn.Tanh() 
            )
        elif architecture == "medium":
            self.network = nn.Sequential(
                nn.Linear(state_size, 256),
                activation_fn,
                nn.Linear(256, 256),
                activation_fn,
                nn.Linear(256, 128),
                activation_fn,
                nn.Linear(128, action_size),
                nn.Tanh()
            )
        elif architecture == "complex":
            self.network = nn.Sequential(
                nn.Linear(state_size, 512),
                activation_fn,
                nn.Linear(512, 512),
                activation_fn,
                nn.Linear(512, 256),
                activation_fn,
                nn.Linear(256, action_size),
                nn.Tanh()
            )
        else:
            raise ValueError("Unknown architecture type. Choose 'simple', 'medium', or 'complex'.")

    def forward(self, state):
        return self.network(state)

class Critic(nn.Module):
    def __init__(self, state_size, action_size, architecture="simple", activation="relu"):
        super(Critic, self).__init__()
        activation_fn = ACTIVATION_FUNCTIONS.get(activation, nn.ReLU())
        input_size = state_size + action_size
        if architecture == "simple":
            self.network = nn.Sequential(
                nn.Linear(input_size, 128),
                activation_fn,
                nn.Linear(128, 128),
                activation_fn,
                nn.Linear(128, 1)
            )
        elif architecture == "medium":
            self.network = nn.Sequential(
                nn.Linear(input_size, 256),
                activation_fn,
                nn.Linear(256, 256),
                activation_fn,
                nn.Linear(256, 128),
                activation_fn,
                nn.Linear(128, 1)
            )
        elif architecture == "complex":
            self.network = nn.Sequential(
                nn.Linear(input_size, 512),
                activation_fn,
                nn.Linear(512, 512),
                activation_fn,
                nn.Linear(512, 256),
                activation_fn,
                nn.Linear(256, 1)
            )
        else:
            raise ValueError("Unknown architecture type. Choose 'simple', 'medium', or 'complex'.")

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.network(x)

class DDPGAgentMultiStock:
    def __init__(self, state_size, action_size, architecture="simple", activation="relu",
                 action_low=-100, action_high=100,
                 actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=1e-3,
                 buffer_size=100000, batch_size=64, device='cpu'):
        
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device(device)
        
        self.actor = Actor(state_size, action_size, architecture=architecture, activation=activation).to(self.device)
        self.target_actor = Actor(state_size, action_size, architecture=architecture, activation=activation).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        self.critic = Critic(state_size, action_size, architecture=architecture, activation=activation).to(self.device)
        self.target_critic = Critic(state_size, action_size, architecture=architecture, activation=activation).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self._soft_update(self.target_actor, self.actor, 1.0)
        self._soft_update(self.target_critic, self.critic, 1.0)
        
        self.memory = deque(maxlen=buffer_size)
        self.noise_std = 0.2
    
        self.loss_history = []
    
    def _soft_update(self, target, source, tau):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau*source_param.data + (1.0 - tau)*target_param.data)
    
    def add_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, add_noise=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        self.actor.train()
        if add_noise:
            noise = np.random.normal(0, self.noise_std, size=self.action_size)
            action += noise
        action = np.clip(action, -1, 1)

        action = action * (self.action_high - self.action_low)/2 + (self.action_high + self.action_low)/2
        return action
    
    def evaluate_agent(self, env, render=False):

        state = env.reset()
        test_portfolios = []
        actions_taken = []
        states = []
        done = False

        while not done:
            action = self.act(state, add_noise=False) 
            next_state, reward, done, info = env.step(action)
            if render:
                env.render()
            portfolio_value = info['portfolio_value']
            if isinstance(portfolio_value, (list, np.ndarray)):
                portfolio_value = float(portfolio_value[0]) if len(portfolio_value) > 0 else 0.0
            else:
                portfolio_value = float(portfolio_value)
            test_portfolios.append(portfolio_value)
            actions_taken.append(action)
            states.append(state)
            state = next_state

        final_portfolio = test_portfolios[-1] if test_portfolios else env.initial_cash
        cumulative_profit_loss = final_portfolio - env.initial_cash
        print(f"Final Portfolio Value: ${final_portfolio:.2f}")
        print(f"Cumulative Profit/Loss: ${cumulative_profit_loss:.2f}")
        return test_portfolios, states, actions_taken

    
    def replay(self):

        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        rewards = [float(r) for r in rewards]
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            Q_targets_next = self.target_critic(next_states, next_actions)
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic(states, actions)
        critic_loss = nn.MSELoss()(Q_expected, Q_targets)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actions_pred = self.actor(states)
        actor_loss = -self.critic(states, actions_pred).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self._soft_update(self.target_actor, self.actor, self.tau)
        self._soft_update(self.target_critic, self.critic, self.tau)
        
        self.loss_history.append({'critic_loss': critic_loss.item(), 'actor_loss': actor_loss.item()})
    
    def compute_metrics(self, portfolio_values):
  
        portfolio_values = [float(pv) for pv in portfolio_values if isinstance(pv, (float, int))]
        
        if len(portfolio_values) < 2:
            print("Not enough portfolio values to compute metrics.")
            return {
                'Cumulative Return': 0.0,
                'Maximum Drawdown': 0.0,
                'Sharpe Ratio': 0.0,
                'Performance Stability': 0.0
            }
        
        portfolio = np.array(portfolio_values, dtype=float)
        
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

    
    def save_learning_animation(self, portfolio_values, filename='learning_animation.gif'):

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(0, len(portfolio_values))
        min_val = min(portfolio_values) - 1000
        max_val = max(portfolio_values) + 1000
        ax.set_ylim(min_val, max_val)
        ax.set_title("Agent's Portfolio Value Over Time")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Portfolio Value (USD)")

        line, = ax.plot([], [], lw=2, label='Portfolio Value')
        portfolio_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')

        def update(frame):
            line.set_data(range(frame), portfolio_values[:frame])
            portfolio_text.set_text(f"Portfolio Value: ${portfolio_values[frame -1]:.2f}")
            return line, portfolio_text

        ani = FuncAnimation(fig, update, frames=len(portfolio_values), blit=True, interval=100)
        ani.save(filename, writer='pillow')
        plt.close()
        print(f"Learning animation saved as {filename}")

    
    def save_loss_animation(self, filename='loss_animation.gif'):

        if not self.loss_history:
            print("No loss history to plot.")
            return
    
        critic_losses = [loss['critic_loss'] for loss in self.loss_history]
        actor_losses = [loss['actor_loss'] for loss in self.loss_history]
    
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(0, len(critic_losses))
        min_val = min(min(critic_losses, default=0), min(actor_losses, default=0))
        max_val = max(max(critic_losses, default=0), max(actor_losses, default=0))
        ax.set_ylim(min_val, max_val)
        ax.set_title("Training Loss Over Time")
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Loss")
        ax.legend(['Critic Loss', 'Actor Loss'])
    
        line1, = ax.plot([], [], lw=2, label='Critic Loss')
        line2, = ax.plot([], [], lw=2, label='Actor Loss')
    
        def update(frame):
            line1.set_data(range(frame), critic_losses[:frame])
            line2.set_data(range(frame), actor_losses[:frame])
            return line1, line2
    
        ani = FuncAnimation(fig, update, frames=len(critic_losses), blit=True, interval=100)
        ani.save(filename, writer='pillow')
        plt.close()
        print(f"Loss animation saved as {filename}")
    
        plt.figure(figsize=(10,6))
        plt.plot(critic_losses, label='Critic Loss')
        plt.plot(actor_losses, label='Actor Loss')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.legend()
        plt.savefig('loss_plot.png')
        plt.close()
    
    def save_model(self, filename='ddpg_model.pth'):

        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename='ddpg_model.pth'):

        checkpoint = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"Model loaded from {filename}")
