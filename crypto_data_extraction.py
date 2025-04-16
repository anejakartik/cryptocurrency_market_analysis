import requests
import pandas as pd
from datetime import datetime

class CryptoDataFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://min-api.cryptocompare.com/data/v2/histoday"
    
    def fetch_historical_data(self, coin, currency="USD", start_date="2023-01-01", end_date="2024-12-31", limit=2000):
        """Fetch historical data for a cryptocurrency from start_date to end_date."""
        # Convert start and end dates to UNIX timestamps
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Convert both dates to UNIX timestamps
        from_ts = int(start_date_obj.timestamp())
        to_ts = int(end_date_obj.timestamp())

        params = {
            'fsym': coin,  # cryptocurrency symbol
            'tsym': currency,  # comparison currency symbol
            'limit': limit,  # max number of data points (2000 max)
            'toTs': to_ts,  # End time for the data
            'api_key': self.api_key  # API key
        }

        response = requests.get(self.base_url, params=params)
        
        if response.status_code != 200:
            raise Exception(f"Error fetching data: {response.status_code}")

        data = response.json()['Data']['Data']

        # Convert the data to a DataFrame
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'], unit='s')  # Convert UNIX timestamp to datetime
        df.set_index('time', inplace=True)
        
        # Filter data between the specified start and end date
        df = df[(df.index >= start_date_obj) & (df.index <= end_date_obj)]
        
        return df

    def save_to_csv(self, df, file_name):
        """Save the fetched data to a CSV file."""
        df.to_csv(file_name)
        print(f"Data saved to {file_name}")

# Example Usage:
if __name__ == "__main__":
    api_key = '89d6ce230a28c1253bd3b2a7ae63d0ee968792d925a9f4368b4671c0f28340b4'  # Replace with your actual API key
    fetcher = CryptoDataFetcher(api_key)
    
    # Fetch data for Bitcoin (BTC) from 2023-01-01 to 2024-12-31
    coins = ['BTC', 'ETH', 'USDT', 'XRP', 'BNB', 'SOL', 'USDC', 'TRX', 'DOGE', 'ADA']
    for coin in coins:
        data = fetcher.fetch_historical_data(coin, start_date="2023-01-01", end_date="2024-12-31")
        fetcher.save_to_csv(data, f'coin_data/{coin}_2023_2024.csv')




