from Rl.environments.EnvironmentYahoo import FinanceEnv
from FinRL.Rl.agents.DQNAgent import DQNAgent

env = FinanceEnv(ticker='AAPL', start_date='2020-01-01', end_date='2021-01-01')

state_size = len(env.data)
action_size = 3  

agent = DQNAgent(state_size, action_size)

num_episodes = 1000
batch_size = 32

for e in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        agent.replay(batch_size)
    if e % 10 == 0:
        agent.target_model.load_state_dict(agent.model.state_dict())
