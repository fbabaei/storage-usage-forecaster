# model_utils.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

def generate_synthetic_storage(days=365, seed=42, trend=0.02, noise_std=5.0, weekly_amp=10.0, monthly_amp=20.0):
    """
    Generate synthetic daily storage usage (GB) with trend + seasonality + noise.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(days)
    # base usage, linear trend, weekly seasonality, monthly seasonality, noise
    base = 500  # starting GB
    trend_component = trend * t  # linear growth
    weekly = weekly_amp * np.sin(2 * np.pi * t / 7.0)
    monthly = monthly_amp * np.sin(2 * np.pi * t / 30.0)
    noise = rng.normal(0, noise_std, size=days)
    usage = base + trend_component + weekly + monthly + noise
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days)
    df = pd.DataFrame({"date": dates, "usage_gb": usage})
    return df

def create_sequences(values, window):
    """
    Create sliding windows for supervised learning.
    values: np.array shape (n,)
    returns X shape (n-window, window, 1), y shape (n-window,)
    """
    X, y = [], []
    for i in range(len(values) - window):
        X.append(values[i:i+window])
        y.append(values[i+window])
    X = np.array(X)
    y = np.array(y)
    # reshape for LSTM (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

def build_lstm(window, hidden_units=64, dropout=0.1):
    model = Sequential()
    model.add(LSTM(hidden_units, input_shape=(window, 1), return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(max(16, hidden_units // 2), return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=16, patience=6, verbose=1):
    es = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=verbose
    )
    return model, history

def forecast_future(model, last_sequence, n_steps, scaler):
    """
    last_sequence: np.array shape (window,) in original scaled space (not necessarily)
    scaler: fitted MinMaxScaler used on training data (expects shape (-1,1))
    Returns predicted values in original units (inverse transformed).
    """
    preds = []
    seq = last_sequence.copy()  # scaled
    for _ in range(n_steps):
        x = seq.reshape((1, seq.shape[0], 1))
        p = model.predict(x, verbose=0)[0,0]  # scaled prediction
        preds.append(p)
        # append and slide
        seq = np.append(seq[1:], p)
    preds = np.array(preds).reshape(-1,1)
    inv = scaler.inverse_transform(preds).flatten()
    return inv

def save_model(model, path="saved_lstm.h5"):
    model.save(path)

def load_saved_model(path="saved_lstm.h5"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return load_model(path)
