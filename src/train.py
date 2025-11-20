import pandas as pd
import argparse
import os
from model import feature_engineer, train_rf, evaluate_model, save_model

def main():
    parser = argparse.ArgumentParser()
    ROOT = os.getcwd()
    parser.add_argument('--data', default='data/energy.csv')
    parser.add_argument('--model-out', default='models/rf_model.pkl')
    args = parser.parse_args()
    print(args.data)
    print(os.getcwd())
    df = pd.read_csv(args.data, parse_dates=['timestamp'], index_col='timestamp')
    df = df.sort_index()
    X = feature_engineer(df)
    y = df['demand'].values

    split_idx = int(len(X)*0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = train_rf(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    os.makedirs(os.path.dirname(args.model_out) or '.', exist_ok=True)
    save_model(model, args.model_out)
    print('Model saved to', args.model_out)
    print('Evaluation metrics:', metrics)

if __name__ == '__main__':
    main()
