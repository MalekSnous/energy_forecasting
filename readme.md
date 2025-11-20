# Energy Forecasting & Resource Optimization

This repository is a starter project for energy forecasting. It contains:
- synthetic data generator (`scripts/generate_synthetic_data.py`)
- training and evaluation pipeline (`src/train.py`, `src/model.py`)
- interactive dashboard (`app/streamlit_app.py`)
- Jupyter analysis script (`notebooks/forecast_pipeline.py`)
- requirements and instructions to run locally or deploy




1. Create and activate a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate    # mac/linux
venv\Scripts\activate       # windows
pip install -r requirements.txt