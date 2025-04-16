import pandas as pd
import glob

class CryptoDataMerger:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.merged_data = None

    def load_and_merge_data(self):
        """Load all CSV files and stack them vertically with a 'coin' identifier"""
        csv_files = glob.glob(f"{self.data_dir}/*.csv")
        dfs = []

        for file in csv_files:
            # Extract the coin name from the file name
            coin_name = file.split("/")[-1].split("_")[0].lower()

            # Load the data
            df = pd.read_csv(file, parse_dates=["time"])

            # Add a column for the coin name
            df["coin"] = coin_name

            # Ensure column order
            df = df[["time", "coin", "open", "high", "low", "close", "volumefrom", "volumeto"]]

            dfs.append(df)

        # Concatenate all DataFrames
        self.merged_data = pd.concat(dfs, ignore_index=True)
        return self.merged_data

    def save_merged_data(self, file_name="merged_crypto_data.csv"):
        if self.merged_data is not None:
            self.merged_data.to_csv(file_name, index=False)
            print(f"Data successfully merged and saved to {file_name}")
        else:
            print("No data to save. Please load and merge data first.")

# Example Usage
if __name__ == "__main__":
    merger = CryptoDataMerger(data_dir="coin_data")
    merged_data = merger.load_and_merge_data()
    merger.save_merged_data(file_name="merged_crypto_2023_2024.csv")
