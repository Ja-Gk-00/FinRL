-


if __name__ == "__main__":
    downloader = StockDataDownloader()
    data = downloader.download_data()

    data_clean = downloader.remove_missing_values(data)
    env = StockTradingEnv(data_clean[['close']])
    agent = QLearningAgent(state_size=len(data_clean.columns), action_size=3)

    train_agent(env, agent)
