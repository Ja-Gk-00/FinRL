
from StockDataDownloader import St






if __name__ == '__main__':
    downloader = StockDataDownloader()
    downloader.download_data()
    downloader.set_ticker('MSFT')
    downloader.set_start_date('2019-01-01')
    downloader.set_end_date('2022-01-01')
    downloader.set_save_path('./data/djia_microsoft')
    downloader.download_data()