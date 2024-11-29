# File: FinRL/train_multistock_agent.py

import os
import pandas as pd
from Rl.environments.MultiStockEnvironment import MultiStockContinuousEnv
from Rl.agents.DDPGAgentMultiStock import DDPGAgentMultiStock

results_folder = "ddpg_multistock_results"
os.makedirs(results_folder, exist_ok=True)

tickers = ['AAPL', 'GOOGL']
train_env = MultiStockContinuousEnv(tickers=tickers, start_date='2020-01-01', end_date='2020-12-31',
                                    initial_cash=100000, max_shares=1000, max_buy=100, max_sell=100)
test_env = MultiStockContinuousEnv(tickers=tickers, start_date='2021-01-01', end_date='2021-12-31',
                                   initial_cash=100000, max_shares=1000, max_buy=100, max_sell=100)

state_size = len(train_env._get_observation())
action_size = len(tickers)

metrics_summary = []

architectures = ["simple", "medium", "complex"]
activation_functions = ["relu", "gelu", "tanh", "leaky_relu"]

for arch in architectures:
    for activation in activation_functions:
        print(f"\nTraining the DDPG Agent ({arch} architecture with {activation} activation)...")
        
        agent = DDPGAgentMultiStock(state_size=state_size, action_size=action_size,
                                   architecture=arch, activation=activation,
                                   action_low=-100, action_high=100,
                                   actor_lr=1e-4, critic_lr=1e-3,
                                   gamma=0.99, tau=1e-3,
                                   buffer_size=100000, batch_size=64,
                                   device='cpu') 
    
        episodes = 500 
        print(f"Starting training for {episodes} episodes...")
        for episode in range(1, episodes + 1):
            state = train_env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action = agent.act(state)
                next_state, reward, done, info = train_env.step(action)
                agent.add_experience(state, action, reward, next_state, done)
                agent.replay()
                state = next_state
                total_reward += reward
            print(f"Episode {episode}/{episodes}, Total Reward: {total_reward:.2f}, Portfolio: ${train_env.portfolio_value:.2f}")
        
        print(f"\nTesting the DDPG Agent ({arch} architecture with {activation} activation)...")
        test_portfolios, states, actions_taken = agent.evaluate_agent(test_env, render=False)
        if len(test_portfolios) == 0:
            print("No portfolio values collected during testing.")
            final_portfolio = test_env.initial_cash
            cumulative_profit_loss = 0.0
        else:
            final_portfolio = test_portfolios[-1]
            cumulative_profit_loss = final_portfolio - test_env.initial_cash
        print(f"Final Portfolio Value: ${final_portfolio:.2f}")
        print(f"Cumulative Profit/Loss: ${cumulative_profit_loss:.2f}")
        
        print(f"Number of portfolio entries: {len(test_portfolios)}")
        print(f"First 5 portfolio values: {test_portfolios[:5]}")
        if len(test_portfolios) > 0:
            print(f"Type of portfolio entries: {type(test_portfolios[0])}")
        
        metrics = agent.compute_metrics(test_portfolios)
        metrics['Architecture'] = arch
        metrics['Activation Function'] = activation
        metrics_summary.append(metrics)
        
        df = pd.DataFrame({'Step': range(len(test_portfolios)), 'Portfolio Value': test_portfolios})
        csv_filename = os.path.join(results_folder, f"ddpg_test_logs_{arch}_{activation}.csv")
        df.to_csv(csv_filename, index=False)
        print(f"Test logs saved to {csv_filename}")
        
        learning_gif_filename = os.path.join(results_folder, f"learning_animation_{arch}_{activation}.gif")
        agent.save_learning_animation(portfolio_values=test_portfolios, filename=learning_gif_filename)
        
        loss_gif_filename = os.path.join(results_folder, f"loss_animation_{arch}_{activation}.gif")
        agent.save_loss_animation(filename=loss_gif_filename)
        
        model_filename = os.path.join(results_folder, f"ddpg_model_{arch}_{activation}.pth")
        agent.save_model(filename=model_filename)
    
metrics_df = pd.DataFrame(metrics_summary)
metrics_csv = os.path.join(results_folder, "performance_metrics_summary.csv")
metrics_df.to_csv(metrics_csv, index=False)
print(f"\nPerformance metrics summary saved to {metrics_csv}")
