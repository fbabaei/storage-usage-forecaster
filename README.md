# 📈 Storage Usage Forecaster (Mini BI + LSTM)

Forecast daily storage usage with an interactive Streamlit dashboard + LSTM model.

## Features
- Synthetic dataset generator (or upload your CSV with `date, usage_gb`)
- Train an LSTM with lookback window
- Visualize historical & forecast usage
- Evaluate RMSE/MAE on test horizon
- Export forecast CSV
- Deployable to Streamlit Cloud / Hugging Face Spaces

## Quickstart
```bash
git clone https://github.com/your-username/storage-forecaster.git
cd storage-forecaster
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
