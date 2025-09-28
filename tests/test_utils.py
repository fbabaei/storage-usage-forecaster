import numpy as np
import pytest
from model_utils import generate_synthetic_storage, create_sequences, build_lstm, forecast_future
from sklearn.preprocessing import MinMaxScaler

def test_generate_synthetic_storage_shape():
    df = generate_synthetic_storage(days=30)
    assert df.shape[0] == 30
    assert "date" in df.columns
    assert "usage_gb" in df.columns

def test_create_sequences_shape():
    arr = np.arange(20)
    X, y = create_sequences(arr, window=5)
    assert X.shape[0] == 15
    assert X.shape[1] == 5
    assert y.shape[0] == 15

def test_forecast_future_runs():
    # simple scaler fit
    arr = np.arange(50).reshape(-1,1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(arr).flatten()

    X, y = create_sequences(scaled, window=10)
    model = build_lstm(window=10, hidden_units=8)
    model.fit(X, y, epochs=1, batch_size=4, verbose=0)

    last_seq = scaled[-10:]
    preds = forecast_future(model, last_seq, n_steps=5, scaler=scaler)
    assert len(preds) == 5
