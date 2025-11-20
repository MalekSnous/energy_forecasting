import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import os

st.set_page_config(layout='wide', page_title='Energy Forecast Dashboard')

DATA_PATH = st.experimental_get_query_params().get('data', ['data/energy.csv'])[0]
MODEL_PATH = st.experimental_get_query_params().get('model', ['models/rf_model.pkl'])[0]

@st.cache_data
def load_data(path):
    return pd.read_csv(path, parse_dates=['timestamp'], index_col='timestamp')

@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

df = load_data(DATA_PATH)
model = load_model(MODEL_PATH)

st.title('Energy Forecasting & Optimization Dashboard')

st.sidebar.header('Controls')
st.sidebar.write('Data & model paths can be set via query params.')

st.header('Historical demand (last 30 days)')
st.line_chart(df['demand'].last('720H'))

st.header('Forecast (next 24 hours)')
last = df.iloc[-48:].copy()
future_index = pd.date_range(start=last.index[-1] + pd.Timedelta(hours=1), periods=24, freq='H')
future = pd.DataFrame(index=future_index)
future['hour'] = future.index.hour
future['dayofweek'] = future.index.dayofweek
future['dayofyear'] = future.index.dayofyear
future['temp_C'] = last['temp_C'].mean()

if model is not None:
    future['sin_hour'] = np.sin(2*np.pi*future['hour']/24)
    future['cos_hour'] = np.cos(2*np.pi*future['hour']/24)
    future['sin_doy'] = np.sin(2*np.pi*future['dayofyear']/365)
    future['cos_doy'] = np.cos(2*np.pi*future['dayofyear']/365)
    Xf = future[['temp_C','hour','dayofweek','sin_hour','cos_hour','sin_doy','cos_doy']]
    preds = model.predict(Xf)
    future['forecast'] = preds
    st.line_chart(pd.concat([df['demand'][-168:], future['forecast']]))
else:
    st.info('Model not found: run training first (src/train.py)')

st.header('Scenario analysis')
st.write('Adjust temperature to see impact on demand forecast.')
temp_adj = st.slider('Temperature adjustment (°C)', -5.0, 5.0, 0.0)
if model is not None:
    Xf2 = Xf.copy()
    Xf2['temp_C'] = Xf2['temp_C'] + temp_adj
    preds2 = model.predict(Xf2)
    future['forecast_adj'] = preds2
    chart_df = pd.concat([df['demand'][-168:], future[['forecast','forecast_adj']]])
    chart_df = chart_df.reset_index().melt('timestamp', var_name='variable', value_name='value')
    chart = alt.Chart(chart_df).mark_line().encode(x='timestamp:T', y='value:Q', color='variable:N').interactive()
    st.altair_chart(chart, use_container_width=True)
