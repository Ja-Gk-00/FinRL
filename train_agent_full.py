from Rl.environments.EnvironmentFinance import FinanceEnvironment
from Rl.agents.DQNAgent import DQNAgent
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

results_folder = "dqn_results"
os.makedirs(results_folder, exist_ok=True)

architectures = ["simple", "medium", "complex"]
activation_functions = ["relu", "gelu", "tanh", "leaky_relu"]
results = {}

train_env = FinanceEnvironment(ticker='AAPL', start_date='2020-01-01', end_date='2020-12-31')
test_env = FinanceEnvironment(ticker='AAPL', start_date='2021-01-01', end_date='2021-12-31')

metrics_summary = []
training_data = {}
testing_data = {}
loss_history = {}

for arch in architectures:
    for activation in activation_functions:
        config_name = f"{arch}_{activation}"
        print(f"\nTraining the DQN Agent ({arch} architecture with {activation} activation)...")
        agent = DQNAgent(state_size=3, architecture=arch, activation=activation, max_action=10)
        agent.train_agent(train_env, episodes=500)
        training_data[config_name] = agent.training_portfolio_values
        print(f"\nTesting the DQN Agent ({arch} architecture with {activation} activation)...")
        test_portfolios, states, actions_taken = agent.evaluate_agent(test_env, render=True)
        testing_data[config_name] = test_portfolios
        metrics = agent.compute_metrics(test_portfolios)
        metrics['Architecture'] = arch
        metrics['Activation Function'] = activation
        metrics_summary.append(metrics)
        df = pd.DataFrame({'Step': range(len(test_portfolios)), 'Portfolio Value': test_portfolios})
        csv_filename = os.path.join(results_folder, f"dqn_test_logs_{arch}_{activation}.csv")
        df.to_csv(csv_filename, index=False)
        print(f"Test logs saved to {csv_filename}")
        loss_history[config_name] = agent.loss_history.copy()
        model_filename = os.path.join(results_folder, f"dqn_model_{arch}_{activation}.pth")
        agent.save_model(filename=model_filename)

fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_title("Training Performance")
ax1.set_xlabel("Episodes")
ax1.set_ylabel("Portfolio Value")
lines1 = {}
for config in training_data:
    lines1[config], = ax1.plot([], [], label=config)
ax1.legend()

def update_train(frame):
    for config, data in training_data.items():
        lines1[config].set_data(range(frame), data[:frame])
    ax1.relim()
    ax1.autoscale_view()
    return list(lines1.values())

ani1 = FuncAnimation(fig1, update_train, frames=max(len(data) for data in training_data.values()),
                     blit=True, interval=100)
ani1.save(os.path.join(results_folder, 'training_performance.gif'), writer='pillow')
plt.close(fig1)

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.set_title("Testing Performance")
ax2.set_xlabel("Steps")
ax2.set_ylabel("Portfolio Value")
lines2 = {}
for config in testing_data:
    lines2[config], = ax2.plot([], [], label=config)
ax2.legend()

def update_test(frame):
    for config, data in testing_data.items():
        lines2[config].set_data(range(frame), data[:frame])
    ax2.relim()
    ax2.autoscale_view()
    return list(lines2.values())

ani2 = FuncAnimation(fig2, update_test, frames=max(len(data) for data in testing_data.values()),
                     blit=True, interval=100)
ani2.save(os.path.join(results_folder, 'testing_performance.gif'), writer='pillow')
plt.close(fig2)

fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.set_title("Training Loss")
ax3.set_xlabel("Iterations")
ax3.set_ylabel("Loss")
lines3 = {}
for config in loss_history:
    lines3[config], = ax3.plot([], [], label=config)
ax3.legend()

def update_loss(frame):
    for config, data in loss_history.items():
        lines3[config].set_data(range(frame), data[:frame])
    ax3.relim()
    ax3.autoscale_view()
    return list(lines3.values())

ani3 = FuncAnimation(fig3, update_loss, frames=max(len(data) for data in loss_history.values()),
                     blit=True, interval=100)
ani3.save(os.path.join(results_folder, 'training_loss.gif'), writer='pillow')
plt.close(fig3)

metrics_df = pd.DataFrame(metrics_summary)
metrics_csv = os.path.join(results_folder, "performance_metrics_summary.csv")
metrics_df.to_csv(metrics_csv, index=False)
print(f"\nPerformance metrics summary saved to {metrics_csv}")
