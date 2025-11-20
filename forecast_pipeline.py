# Forecast pipeline analysis (script-style, suitable for conversion to notebook)
import pandas as pd
import numpy as np
from src.model import feature_engineer, train_rf, evaluate_model
import matplotlib.pyplot as plt

df = pd.read_csv('data/energy.csv', parse_dates=['timestamp'], index_col='timestamp')
df = df.sort_index()
print(df.head())

X = feature_engineer(df)
y = df['demand'].values

split_idx = int(len(X)*0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

model = train_rf(X_train, y_train)
metrics = evaluate_model(model, X_test, y_test)
print('Eval:', metrics)

plt.figure(figsize=(12,5))
plt.plot(df['demand'][-240:], label='demand')
plt.legend()
plt.show()
