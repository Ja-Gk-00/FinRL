from Rl.environments.EnvironmentYahoo import FinanceEnv
from Rl.agents.RandomAgent import RandomAgent
from VisTools.vis_func import UtilsVis

#Script runs random agent for given data and saves the logs to given dir

# Env
env = FinanceEnv(ticker='AAPL', start_date='2020-01-01', end_date='2021-01-01')

state_size = len(env.data)
action_size = 3  # 0: Buy, 1: Sell, 2: Hold, idk czy tak uzgodnilismy ale tak bedzie tu xdd

random_agent = RandomAgent(state_size, action_size)
# Training
print("\nTraining Random Agent...")
random_agent.train_agent(env, episodes=100)
random_agent.save_logs('random_agent_logs.csv')


UtilsVis.visualize_agent_logs('random_agent_logs.csv', 'RandomAgent')


# To tworzy gifa, idk nie działa mi na kompie (zacina się w trakcie) xddd, do prezki zrobiłem po prostu w jupyterze z danych
# UtilsVis.create_portfolio_gif(
#     log_filename='random_agent_logs.csv',
#     gif_filename='random_agent_portfolio.gif',
#     agent_name='RandomAgent'
# )