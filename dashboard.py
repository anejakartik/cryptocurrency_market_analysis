import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import datetime
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# -----------------------------
# CryptoCompare API Fetcher
# -----------------------------
class CryptoDataFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://min-api.cryptocompare.com/data/v2/histoday"

    def fetch_historical_data(self, coin, currency="USD", days=2000, limit=2000):
        end_date_obj = datetime.datetime.utcnow()
        to_ts = int(end_date_obj.timestamp())

        params = {
            'fsym': coin,
            'tsym': currency,
            'limit': limit,
            'toTs': to_ts,
            'api_key': self.api_key
        }

        response = requests.get(self.base_url, params=params)
        if response.status_code != 200:
            raise Exception(f"Error fetching data: {response.status_code}")

        data = response.json()['Data']['Data']
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df

# -----------------------------
# Helper: Add indicators
# -----------------------------
def create_lagged_features(group):
    group['daily_return'] = group['close'].pct_change()
    for lag in [1, 3, 7]:
        group[f'return_lag_{lag}'] = group['daily_return'].shift(lag)
    group['volume_ma_7'] = group['volumeto'].rolling(7).mean()
    group['volume_zscore'] = (group['volumeto'] - group['volume_ma_7']) / group['volumeto'].std()
    return group

def add_indicators(df):
    df = df.copy()
    df['rsi_14'] = RSIIndicator(close=df['close'], window=14).rsi()
    macd = MACD(close=df['close'])
    df['macd_line'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    bb = BollingerBands(close=df['close'])
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['vwap'] = VolumeWeightedAveragePrice(
        high=df['high'], low=df['low'], close=df['close'], volume=df['volumeto'], window=14
    ).volume_weighted_average_price()
    df = df.groupby('coin', group_keys=False).apply(create_lagged_features)
    df['coin_code'] = df['coin'].astype('category').cat.codes
    df = df.dropna()
    return df

# -----------------------------
# Load models
# -----------------------------
rf_model = joblib.load('models/rf_model.pkl')
xgb_model = joblib.load('models/xgb_model.pkl')
lr_model = joblib.load('models/lr_model.pkl')
dt_model = joblib.load('models/dt_model.pkl')
lstm_model = load_model('models/lstm_model.h5')
scaler = joblib.load('models/lstm_scaler.pkl')
coin_encoder = joblib.load('models/coin_encoder.pkl')
lstm_config = joblib.load('models/lstm_config.pkl')

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Crypto Trend Prediction", layout="wide")
st.title("📈 Live Crypto Trend Prediction Dashboard")

api_key = '89d6ce230a28c1253bd3b2a7ae63d0ee968792d925a9f4368b4671c0f28340b4'
fetcher = CryptoDataFetcher(api_key)

coin_map = {
    'Bitcoin': 'BTC', 'Ethereum': 'ETH', 'Tether': 'USDT', 'Ripple': 'XRP',
    'BNB': 'BNB', 'Solana': 'SOL', 'USD Coin': 'USDC', 'TRON': 'TRX',
    'Dogecoin': 'DOGE', 'Cardano': 'ADA'
}

selected_coin_name = st.selectbox("Select Cryptocurrency", list(coin_map.keys()))
selected_coin_symbol = coin_map[selected_coin_name]

# Date Range Filter
today = datetime.date.today()
min_start = today - datetime.timedelta(days=2000)
start_date = st.date_input("Start Date", value=today - datetime.timedelta(days=60), min_value=min_start, max_value=today)
end_date = st.date_input("End Date", value=today, min_value=start_date, max_value=today)

# Fetch and combine data for all coins
all_data = []
for name, symbol in coin_map.items():
    df_coin = fetcher.fetch_historical_data(symbol)
    df_coin['coin'] = symbol.lower()
    all_data.append(df_coin)

df = pd.concat(all_data)
df = add_indicators(df)
df = df[df['coin'] == selected_coin_symbol.lower()]
df = df[(df.index.date >= start_date) & (df.index.date <= end_date)]

st.line_chart(df['close'])

# One-hot encode coin column for logistic regression
df_encoded = pd.get_dummies(df, columns=['coin'], prefix='coin')
expected_cols = [f'coin_{c.lower()}' for c in ['btc', 'eth', 'usdt', 'xrp', 'bnb', 'sol', 'usdc', 'trx', 'doge', 'ada']]
for col in expected_cols:
    if col not in df_encoded.columns:
        df_encoded[col] = 0

# Model-specific features
base_features = ['rsi_14', 'daily_return', 'volumeto', 'vwap', 'coin_code']
lr_features = ['rsi_14', 'volumeto', 'macd_line'] + expected_cols

dt_features = base_features + ['bb_upper', 'bb_lower', 'volume_zscore']
rf_features = base_features + ['return_lag_1', 'return_lag_3', 'macd_line']
xgb_features = base_features + ['return_lag_1', 'return_lag_3', 'return_lag_7', 'bb_upper', 'bb_lower']
lstm_features = ['close', 'volumeto', 'rsi_14', 'macd_line']

# Evaluate models
def evaluate_predictions(y_true, y_pred):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred)
    }

results = {}
df['actual_movement'] = (df['close'].shift(-1) > df['close']).astype(int)

# Logistic Regression
df['lr_prediction'] = lr_model.predict(df_encoded[lr_features])
results['Logistic Regression'] = evaluate_predictions(df['actual_movement'].dropna(), df['lr_prediction'].dropna())

# Decision Tree
df['dt_prediction'] = dt_model.predict(df[dt_features])
results['Decision Tree'] = evaluate_predictions(df['actual_movement'].dropna(), df['dt_prediction'].dropna())

# Random Forest
df['rf_prediction'] = rf_model.predict(df[rf_features])
results['Random Forest'] = evaluate_predictions(df['actual_movement'].dropna(), df['rf_prediction'].dropna())

# XGBoost
df['xgb_prediction'] = xgb_model.predict(df[xgb_features])
results['XGBoost'] = evaluate_predictions(df['actual_movement'].dropna(), df['xgb_prediction'].dropna())

# LSTM
lstm_scaled = scaler.transform(df[lstm_features])
lstm_coin_encoded = coin_encoder.transform(df['coin'])
X_seq = []
X_coin = []
sequence_length = lstm_config['sequence_length']
for i in range(len(lstm_scaled) - sequence_length):
    X_seq.append(lstm_scaled[i:i + sequence_length])
    X_coin.append(lstm_coin_encoded[i:i + sequence_length])

X_seq = np.array(X_seq)
X_coin = np.array(X_coin)
lstm_proba = lstm_model.predict([X_seq, X_coin]).flatten()
lstm_pred = (lstm_proba > 0.5).astype(int)
df['lstm_prediction'] = np.nan
df.loc[df.index[-len(lstm_pred):], 'lstm_prediction'] = lstm_pred
lstm_true = df['actual_movement'].iloc[-len(lstm_pred):]
results['LSTM'] = evaluate_predictions(lstm_true, lstm_pred)

# Display comparison table
st.subheader("📊 Model Performance Comparison")
performance_df = pd.DataFrame(results).T
st.dataframe(performance_df.style.format("{:.2%}"))

# Prediction Curve Selector
model_colors = {
    'Logistic Regression': 'blue',
    'Decision Tree': 'purple',
    'Random Forest': 'green',
    'XGBoost': 'orange',
    'LSTM': 'red'
}
selected_models = st.multiselect(
    "Select models to display on chart:",
    options=list(model_colors.keys()),
    default=['Random Forest']
)

# Plot predictions
st.subheader("📈 Price and Prediction Overlay")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, df['close'], label='Actual Price', color='black', linewidth=2)

if 'Logistic Regression' in selected_models:
    ax.plot(df.index, df['lr_prediction'] * df['close'], label='Logistic Regression', color=model_colors['Logistic Regression'], linestyle='--')
if 'Decision Tree' in selected_models:
    ax.plot(df.index, df['dt_prediction'] * df['close'], label='Decision Tree', color=model_colors['Decision Tree'], linestyle='--')
if 'Random Forest' in selected_models:
    ax.plot(df.index, df['rf_prediction'] * df['close'], label='Random Forest', color=model_colors['Random Forest'], linestyle='--')
if 'XGBoost' in selected_models:
    ax.plot(df.index, df['xgb_prediction'] * df['close'], label='XGBoost', color=model_colors['XGBoost'], linestyle='--')
if 'LSTM' in selected_models:
    ax.plot(df.index[-len(lstm_pred):], lstm_pred * df['close'].iloc[-len(lstm_pred):], label='LSTM', color=model_colors['LSTM'], linestyle='--')

ax.set_title(f"Price and Predictions for {selected_coin_name}", fontsize=14)
ax.legend()
ax.grid(True)
st.pyplot(fig)

# import streamlit as st
# import pandas as pd
# import numpy as np
# import requests
# import joblib
# import datetime
# from ta.momentum import RSIIndicator
# from ta.trend import MACD
# from ta.volatility import BollingerBands
# from ta.volume import VolumeWeightedAveragePrice
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from tensorflow.keras.models import load_model
# import matplotlib.pyplot as plt

# # -----------------------------
# # CryptoCompare API Fetcher
# # -----------------------------
# class CryptoDataFetcher:
#     def __init__(self, api_key):
#         self.api_key = api_key
#         self.base_url = "https://min-api.cryptocompare.com/data/v2/histoday"

#     def fetch_historical_data(self, coin, currency="USD", days=2000, limit=2000):
#         end_date_obj = datetime.datetime.utcnow()
#         to_ts = int(end_date_obj.timestamp())

#         params = {
#             'fsym': coin,
#             'tsym': currency,
#             'limit': limit,
#             'toTs': to_ts,
#             'api_key': self.api_key
#         }

#         response = requests.get(self.base_url, params=params)
#         if response.status_code != 200:
#             raise Exception(f"Error fetching data: {response.status_code}")

#         data = response.json()['Data']['Data']
#         df = pd.DataFrame(data)
#         df['time'] = pd.to_datetime(df['time'], unit='s')
#         df.set_index('time', inplace=True)
#         return df

# # -----------------------------
# # Helper: Add indicators
# # -----------------------------
# def create_lagged_features(group):
#     group['daily_return'] = group['close'].pct_change()
#     for lag in [1, 3, 7]:
#         group[f'return_lag_{lag}'] = group['daily_return'].shift(lag)
#     group['volume_ma_7'] = group['volumeto'].rolling(7).mean()
#     group['volume_zscore'] = (group['volumeto'] - group['volume_ma_7']) / group['volumeto'].std()
#     return group

# def add_indicators(df):
#     df = df.copy()
#     df['rsi_14'] = RSIIndicator(close=df['close'], window=14).rsi()
#     macd = MACD(close=df['close'])
#     df['macd_line'] = macd.macd()
#     df['macd_signal'] = macd.macd_signal()
#     bb = BollingerBands(close=df['close'])
#     df['bb_upper'] = bb.bollinger_hband()
#     df['bb_lower'] = bb.bollinger_lband()
#     df['vwap'] = VolumeWeightedAveragePrice(
#         high=df['high'], low=df['low'], close=df['close'], volume=df['volumeto'], window=14
#     ).volume_weighted_average_price()
#     df = df.groupby('coin', group_keys=False).apply(create_lagged_features)
#     df['coin_code'] = df['coin'].astype('category').cat.codes
#     df = df.dropna()
#     return df

# # -----------------------------
# # Load models
# # -----------------------------
# rf_model = joblib.load('models/rf_model.pkl')
# xgb_model = joblib.load('models/xgb_model.pkl')
# lr_model = joblib.load('models/lr_model.pkl')
# dt_model = joblib.load('models/dt_model.pkl')
# lstm_model = load_model('models/lstm_model.h5')
# scaler = joblib.load('models/lstm_scaler.pkl')
# coin_encoder = joblib.load('models/coin_encoder.pkl')
# lstm_config = joblib.load('models/lstm_config.pkl')

# # -----------------------------
# # Streamlit UI
# # -----------------------------
# st.set_page_config(page_title="Crypto Trend Prediction", layout="wide")
# st.title("📈 Live Crypto Trend Prediction Dashboard")

# api_key = '89d6ce230a28c1253bd3b2a7ae63d0ee968792d925a9f4368b4671c0f28340b4'
# fetcher = CryptoDataFetcher(api_key)

# coin_map = {
#     'Bitcoin': 'BTC', 'Ethereum': 'ETH', 'Tether': 'USDT', 'Ripple': 'XRP',
#     'BNB': 'BNB', 'Solana': 'SOL', 'USD Coin': 'USDC', 'TRON': 'TRX',
#     'Dogecoin': 'DOGE', 'Cardano': 'ADA'
# }

# selected_coin_name = st.selectbox("Select Cryptocurrency", list(coin_map.keys()))
# selected_coin_symbol = coin_map[selected_coin_name]

# # Date Range Filter
# today = datetime.date.today()
# min_start = today - datetime.timedelta(days=2000)
# start_date = st.date_input("Start Date", value=today - datetime.timedelta(days=60), min_value=min_start, max_value=today)
# end_date = st.date_input("End Date", value=today, min_value=start_date, max_value=today)

# # Fetch and combine data for all coins
# all_data = []
# for name, symbol in coin_map.items():
#     df_coin = fetcher.fetch_historical_data(symbol)
#     df_coin['coin'] = symbol.lower()
#     all_data.append(df_coin)

# df = pd.concat(all_data)
# df = add_indicators(df)
# df = df[df['coin'] == selected_coin_symbol.lower()]
# df = df[(df.index.date >= start_date) & (df.index.date <= end_date)]

# st.line_chart(df['close'])

# # One-hot encode coin column for logistic regression
# df_encoded = pd.get_dummies(df, columns=['coin'], prefix='coin')
# expected_cols = [f'coin_{c.lower()}' for c in ['btc', 'eth', 'usdt', 'xrp', 'bnb', 'sol', 'usdc', 'trx', 'doge', 'ada']]
# for col in expected_cols:
#     if col not in df_encoded.columns:
#         df_encoded[col] = 0

# # Model-specific features
# base_features = ['rsi_14', 'daily_return', 'volumeto', 'vwap', 'coin_code']
# lr_features = ['rsi_14', 'volumeto', 'macd_line'] + expected_cols

# dt_features = base_features + ['bb_upper', 'bb_lower', 'volume_zscore']
# rf_features = base_features + ['return_lag_1', 'return_lag_3', 'macd_line']
# xgb_features = base_features + ['return_lag_1', 'return_lag_3', 'return_lag_7', 'bb_upper', 'bb_lower']
# lstm_features = ['close', 'volumeto', 'rsi_14', 'macd_line']

# # Evaluate models
# def evaluate_predictions(y_true, y_pred):
#     return {
#         'Accuracy': accuracy_score(y_true, y_pred),
#         'Precision': precision_score(y_true, y_pred),
#         'Recall': recall_score(y_true, y_pred),
#         'F1 Score': f1_score(y_true, y_pred)
#     }

# results = {}
# df['actual_movement'] = (df['close'].shift(-1) > df['close']).astype(int)

# # Logistic Regression
# df['lr_prediction'] = lr_model.predict(df_encoded[lr_features])
# results['Logistic Regression'] = evaluate_predictions(df['actual_movement'].dropna(), df['lr_prediction'].dropna())

# # Decision Tree
# df['dt_prediction'] = dt_model.predict(df[dt_features])
# results['Decision Tree'] = evaluate_predictions(df['actual_movement'].dropna(), df['dt_prediction'].dropna())

# # Random Forest
# df['rf_prediction'] = rf_model.predict(df[rf_features])
# results['Random Forest'] = evaluate_predictions(df['actual_movement'].dropna(), df['rf_prediction'].dropna())

# # XGBoost
# df['xgb_prediction'] = xgb_model.predict(df[xgb_features])
# results['XGBoost'] = evaluate_predictions(df['actual_movement'].dropna(), df['xgb_prediction'].dropna())

# # LSTM
# lstm_scaled = scaler.transform(df[lstm_features])
# lstm_coin_encoded = coin_encoder.transform(df['coin'])
# X_seq = []
# X_coin = []
# sequence_length = lstm_config['sequence_length']
# for i in range(len(lstm_scaled) - sequence_length):
#     X_seq.append(lstm_scaled[i:i + sequence_length])
#     X_coin.append(lstm_coin_encoded[i:i + sequence_length])

# X_seq = np.array(X_seq)
# X_coin = np.array(X_coin)
# lstm_proba = lstm_model.predict([X_seq, X_coin]).flatten()
# lstm_pred = (lstm_proba > 0.5).astype(int)
# df['lstm_prediction'] = np.nan
# df.loc[df.index[-len(lstm_pred):], 'lstm_prediction'] = lstm_pred
# lstm_true = df['actual_movement'].iloc[-len(lstm_pred):]
# results['LSTM'] = evaluate_predictions(lstm_true, lstm_pred)

# # Display comparison table
# st.subheader("📊 Model Performance Comparison")
# performance_df = pd.DataFrame(results).T
# st.dataframe(performance_df.style.format("{:.2%}"))