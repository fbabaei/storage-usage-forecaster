# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
from model_utils import generate_synthetic_storage, create_sequences, build_lstm, train_model, forecast_future, save_model, load_saved_model

st.set_page_config(page_title="Storage Usage Forecaster", layout="wide")
st.title("📈 Storage Usage Forecaster — Mini BI + LSTM")

# Sidebar controls
st.sidebar.header("Settings")
use_sample = st.sidebar.checkbox("Use synthetic sample data", value=True)
uploaded_file = st.sidebar.file_uploader("Or upload CSV (date,usage_gb)", type=["csv"])
window = st.sidebar.slider("Window size (days)", 7, 60, 30)
test_size_days = st.sidebar.slider("Forecast horizon / test days", 7, 90, 30)
epochs = st.sidebar.slider("Training epochs", 5, 200, 50)
hidden_units = st.sidebar.slider("LSTM units (first layer)", 8, 256, 64)
retrain = st.sidebar.button("Retrain model")

# Load data
if use_sample or uploaded_file is None:
    df = generate_synthetic_storage(days=365)
else:
    df = pd.read_csv(uploaded_file, parse_dates=["date"])
    df = df.sort_values("date")
    if "usage_gb" not in df.columns:
        st.error("CSV must contain 'usage_gb' column.")
        st.stop()

st.subheader("Data preview")
st.dataframe(df.tail(10))

st.subheader("Usage over time")
fig = px.line(df, x="date", y="usage_gb", title="Daily Storage Usage (GB)")
st.plotly_chart(fig, use_container_width=True)

# Prepare data
values = df["usage_gb"].values.reshape(-1,1)
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(values).flatten()

# train/test split (last test_size_days reserved)
n_test = test_size_days
n_total = len(scaled)
if n_test + window >= n_total:
    st.error("Increase dataset length or decrease window/test horizon.")
    st.stop()

train_scaled = scaled[:n_total - n_test]
test_scaled = scaled[n_total - n_test - window:]  # include the last window before test

# create sequences
X_train, y_train = create_sequences(train_scaled, window)
# For validation, create sequences that include part of the end of training (simple split)
split_idx = int(0.9 * len(X_train))
X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

st.write(f"Training samples: {len(X_tr)}  Validation samples: {len(X_val)}  Test horizon: {n_test} days")

# Build or load model
MODEL_PATH = "saved_lstm.h5"
model = None
if not retrain and st.sidebar.checkbox("Use saved model if exists", value=True):
    try:
        model = load_saved_model(MODEL_PATH)
        st.sidebar.success("Loaded saved model.")
    except Exception:
        st.sidebar.info("No saved model found; will train a new one.")

if model is None:
    if st.sidebar.button("Build model"):
        model = build_lstm(window, hidden_units=hidden_units)
        st.sidebar.success("Model built (in memory).")
    else:
        # build silently
        model = build_lstm(window, hidden_units=hidden_units)

if retrain or model is None:
    st.info("Training model...")
    model, history = train_model(model, X_tr, y_tr, X_val, y_val, epochs=epochs, batch_size=16)
    save_model(model, MODEL_PATH)
    st.success("Training complete and model saved.")

# Forecast: rolling prediction over test horizon
# Prepare the last sequence from scaled (the last 'window' starting at end of train)
last_seq = scaled[n_total - n_test - window : n_total - n_test]
preds_inv = forecast_future(model, last_seq, n_steps=n_test, scaler=scaler)

# Build timeline for preds
pred_dates = pd.date_range(start=df["date"].iloc[-n_test], periods=n_test)
df_pred = pd.DataFrame({"date": pred_dates, "pred_usage_gb": preds_inv})

# For comparison, if real test values exist, show them
if len(df) >= n_test:
    actuals = df["usage_gb"].iloc[-n_test:].values
    df_pred["actual_usage_gb"] = actuals
    rmse = math.sqrt(mean_squared_error(actuals, preds_inv))
    mae = np.mean(np.abs(actuals - preds_inv))
    st.metric("RMSE (test horizon)", f"{rmse:.2f} GB")
    st.metric("MAE (test horizon)", f"{mae:.2f} GB")

st.subheader("Forecast vs Actual")
fig2 = px.line()
fig2.add_scatter(x=df["date"].iloc[-(n_test*2):], y=df["usage_gb"].iloc[-(n_test*2):], mode="lines", name="Historical")
fig2.add_scatter(x=df_pred["date"], y=df_pred["pred_usage_gb"], mode="lines+markers", name="Forecast")
if "actual_usage_gb" in df_pred.columns:
    fig2.add_scatter(x=df_pred["date"], y=df_pred["actual_usage_gb"], mode="markers", name="Actual")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Forecast numeric")
st.dataframe(df_pred)

# Provide next N days forecast export
st.markdown("### Export forecast")
if st.button("Download forecast CSV"):
    csv = df_pred.to_csv(index=False)
    st.download_button("Click to download CSV", data=csv, file_name="forecast.csv", mime="text/csv")

st.markdown("---")
st.caption("Built with LSTM (TensorFlow/Keras). Model trained on scaled data (MinMaxScaler).")
