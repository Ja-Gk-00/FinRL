# File: FinRL/train_agent.py

from Rl.environments.EnvironmentFinance import FinanceEnvironment
from Rl.agents.DQNAgent import DQNAgent
import os
import pandas as pd

results_folder = "dqn_results"
os.makedirs(results_folder, exist_ok=True)

architectures = ["simple", "medium", "complex"]
activation_functions = ["relu", "gelu", "tanh", "leaky_relu"]
results = {}

train_env = FinanceEnvironment(ticker='AAPL', start_date='2020-01-01', end_date='2020-12-31')
test_env = FinanceEnvironment(ticker='AAPL', start_date='2021-01-01', end_date='2021-12-31')

metrics_summary = []

for arch in architectures:
    for activation in activation_functions:
        print(f"\nTraining the DQN Agent ({arch} architecture with {activation} activation)...")
        agent = DQNAgent(state_size=3, architecture=arch, activation=activation, max_action=10)
        agent.train_agent(train_env, episodes=500)

        print(f"\nTesting the DQN Agent ({arch} architecture with {activation} activation)...")

        test_portfolios, states, actions_taken = agent.evaluate_agent(test_env, render=True)
        results[f"{arch}_{activation}"] = test_portfolios

        metrics = agent.compute_metrics(test_portfolios)
        metrics['Architecture'] = arch
        metrics['Activation Function'] = activation
        metrics_summary.append(metrics)

        df = pd.DataFrame({'Step': range(len(test_portfolios)), 'Portfolio Value': test_portfolios})
        csv_filename = os.path.join(results_folder, f"dqn_test_logs_{arch}_{activation}.csv")
        df.to_csv(csv_filename, index=False)
        print(f"Test logs saved to {csv_filename}")

        learning_gif_filename = os.path.join(results_folder, f"learning_animation_{arch}_{activation}.gif")
        agent.save_learning_animation(states=states, actions=actions_taken, portfolio_values=test_portfolios, filename=learning_gif_filename)

        loss_gif_filename = os.path.join(results_folder, f"loss_animation_{arch}_{activation}.gif")
        agent.plot_loss(filename=loss_gif_filename)

        model_filename = os.path.join(results_folder, f"dqn_model_{arch}_{activation}.pth")
        agent.save_model(filename=model_filename)

metrics_df = pd.DataFrame(metrics_summary)
metrics_csv = os.path.join(results_folder, "performance_metrics_summary.csv")
metrics_df.to_csv(metrics_csv, index=False)
print(f"\nPerformance metrics summary saved to {metrics_csv}")
