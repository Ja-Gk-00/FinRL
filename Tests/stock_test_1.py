if __name__ == "__main__":
    # Przygotowanie danych
    downloader = StockDataDownloader()
    data = downloader.download_data()

    # Usunięcie brakujących wartości
    data_clean = downloader.remove_missing_values(data)

    # Inicjalizacja środowiska i agenta
    env = StockTradingEnv(data_clean[['close']])
    agent = QLearningAgent(state_size=len(data_clean.columns), action_size=3)

    # Trening agenta
    train_agent(env, agent)
