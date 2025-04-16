import requests
import pandas as pd
import praw
from datetime import datetime, timedelta
from textblob import TextBlob
from pathlib import Path
import os
class CryptoDataProcessor:
    def __init__(self, coingecko_params=None, reddit_credentials=None, sql_connection=None):
        """
        Initialize with API credentials and database connection
        
        Parameters:
        - coingecko_params: dict {'coin_id': 'bitcoin', 'days': '365', 'currency': 'usd'}
        - reddit_credentials: dict {'client_id': '', 'client_secret': '', 'user_agent': ''}
        - sql_connection: dict {'server': '', 'database': '', 'username': '', 'password': ''}
        """
        self.coingecko_params = coingecko_params or {
            'coin_id': 'bitcoin',
            'days': '365',
            'currency': 'usd'
        }
        
        self.reddit_credentials = reddit_credentials
        self.sql_connection = sql_connection
        
        # Initialize data containers
        self.crypto_data = None
        self.reddit_data = None
        
    def fetch_crypto_data(self):
        """Fetch cryptocurrency data from CoinGecko API"""
        coin_id = self.coingecko_params['coin_id']
        days = self.coingecko_params['days']
        currency = self.coingecko_params['currency']
        
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency={currency}&days={days}"
        response = requests.get(url)
        data = response.json()
        
        # Convert to DataFrames
        prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
        market_caps = pd.DataFrame(data["market_caps"], columns=["timestamp", "market_cap"])
        volumes = pd.DataFrame(data["total_volumes"], columns=["timestamp", "volume"])
        
        # Merge and clean
        df = prices.merge(market_caps, on="timestamp").merge(volumes, on="timestamp")
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.drop("timestamp", axis=1, inplace=True)
        
        self.crypto_data = df
        return df
    
    def fetch_crypto_data_for_2024(self):
        """Fetch 2024 cryptocurrency data with 365-day API limit workaround"""
        coin_id = self.coingecko_params['coin_id']
        currency = self.coingecko_params['currency']
        
        # Calculate dynamic date ranges that comply with 365-day limit
        today = datetime.now()
        if today.year >= 2025:
            # If current year is 2025+, we can only get partial 2024 data
            start_date = max(
                datetime(2024, 1, 1),
                today - timedelta(days=364)  # 365 days back from today
            )
        else:
            # If running in 2024, we can get full year-to-date data
            start_date = datetime(2024, 1, 1)
        
        end_date = min(
            datetime(2024, 12, 31),
            today - timedelta(days=1)  # Yesterday to ensure data availability
        )
        
        start_ts = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())
        
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range?vs_currency={currency}&from={start_ts}&to={end_ts}"
        response = requests.get(url)
        
        if response.status_code != 200:
            raise ValueError(f"API Error: {response.status_code} - {response.text}")
        
        data = response.json()
        
        # Process data
        prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
        market_caps = pd.DataFrame(data["market_caps"], columns=["timestamp", "market_cap"])
        volumes = pd.DataFrame(data["total_volumes"], columns=["timestamp", "volume"])
        
        # Merge and clean
        df = prices.merge(market_caps, on="timestamp").merge(volumes, on="timestamp")
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.drop("timestamp", axis=1, inplace=True)
        
        # Calculate coverage
        actual_days = df["date"].dt.date.nunique()
        expected_days = (end_date - start_date).days + 1
        coverage = actual_days / expected_days
        
        print(f"Fetched {actual_days} days of data ({coverage:.1%} of requested range)")
        print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        
        if coverage < 0.9:
            print("Warning: Significant data gaps detected")
        
        self.crypto_data = df
        return df
    def fetch_reddit_data(self, subreddit="CryptoCurrency", limit=100):
        """Fetch Reddit posts and calculate sentiment scores"""
        if not self.reddit_credentials:
            raise ValueError("Reddit credentials not provided")
            
        reddit = praw.Reddit(
            client_id=self.reddit_credentials['client_id'],
            client_secret=self.reddit_credentials['client_secret'],
            user_agent=self.reddit_credentials['user_agent']
        )
        
        posts = []
        for post in reddit.subreddit(subreddit).hot(limit=limit):
            sentiment = TextBlob(post.title).sentiment.polarity
            
            posts.append({
                "title": post.title,
                "score": post.score,
                "upvote_ratio": post.upvote_ratio,
                "num_comments": post.num_comments,
                "created_utc": datetime.utcfromtimestamp(post.created_utc),
                "sentiment": sentiment
            })
        
        self.reddit_data = pd.DataFrame(posts)
        return self.reddit_data
    
    def fetch_reddit_data_for_year(self, subreddit="CryptoCurrency", year=2024, limit=1000):
        """Fetch Reddit posts from a specific year"""
        if not self.reddit_credentials:
            raise ValueError("Reddit credentials not configured")

        reddit = praw.Reddit(**self.reddit_credentials)
        posts = []
        
        for post in reddit.subreddit(subreddit).top(limit=limit):  # or .hot()/.new()
            post_date = datetime.utcfromtimestamp(post.created_utc)
            if post_date.year == year:  # Filter for 2024 only
                sentiment = TextBlob(post.title).sentiment.polarity
                posts.append({
                    "title": post.title,
                    "score": post.score,
                    "upvote_ratio": post.upvote_ratio,
                    "num_comments": post.num_comments,
                    "created_utc": post_date,
                    "sentiment": sentiment,
                    "post_url": f"https://reddit.com{post.permalink}"
                })
        
        self.reddit_data = pd.DataFrame(posts)
        return self.reddit_data
    
    def save_to_csv(self, data_type="both", file_prefix="crypto_data", output_dir="data"):
        """
        Save data to CSV files in a specified directory (creates dir if needed)
        
        Parameters:
        - data_type: 'crypto', 'reddit', or 'both'
        - file_prefix: Prefix for filenames (default: 'crypto_data')
        - output_dir: Directory to save files (default: 'data/')
        """
        try:
            # Create directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            if data_type in ("crypto", "both") and self.crypto_data is not None:
                file_path = os.path.join(output_dir, f"{file_prefix}_crypto.csv")
                self.crypto_data.to_csv(file_path, index=False)
                print(f"Saved crypto data to: {file_path}")
            
            if data_type in ("reddit", "both") and self.reddit_data is not None:
                file_path = os.path.join(output_dir, f"{file_prefix}_reddit.csv")
                self.reddit_data.to_csv(file_path, index=False)
                print(f"Saved Reddit data to: {file_path}")
                
        except Exception as e:
            print(f"Error saving CSV files: {str(e)}")
            raise



# Example Usage
if __name__ == "__main__":
    # Initialize with your credentials
    processor = CryptoDataProcessor(
        coingecko_params={'coin_id': 'bitcoin', 'days': '365', 'currency': 'usd'},
        reddit_credentials={
            'client_id': 'ZzynVHSDuFrpNdbpectnow',
            'client_secret': 'Xg5BXiQb3BiQYrxg2R9TqA5aoBrUjw',
            'user_agent': 'windows:sentiment_analysis:v1.0 (by /u/anejjakartik)'
        }
    )
    
    # # Fetch data
    # crypto_df = processor.fetch_crypto_data()
    # reddit_df = processor.fetch_reddit_data(limit=50)
    
    # # Save to CSV
    # processor.save_to_csv(file_prefix="april_2024_data")
    # 1. Fetch data
    # processor.fetch_reddit_data_for_year(year=2024, limit=2000)
    processor.fetch_crypto_data_for_2024()  # 2024 prices

    # # 2. Merge and analyze
    # merged = pd.merge(
    #     left=processor.crypto_data,
    #     right=processor.reddit_data.groupby(processor.reddit_data["created_utc"].dt.date)["sentiment"].mean(),
    #     left_on=processor.crypto_data["date"].dt.date,
    #     right_index=True
    # )

    # 3. Save everything
    processor.save_to_csv(
        data_type="crypto",
        file_prefix="full_2024_analysis"
    )

