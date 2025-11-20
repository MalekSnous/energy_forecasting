###generate_synthetic_data.py


#!/usr/bin/env python3
"""Generate synthetic hourly electricity demand data with weather influence.
Usage: python scripts/generate_synthetic_data.py --out data/energy.csv --days 730
"""
import numpy as np
import pandas as pd
import argparse
import os

def generate(hours=24*365, seed=0):
    np.random.seed(seed)
    rng = pd.date_range(end=pd.Timestamp.today().normalize(), periods=hours, freq='H')
    df = pd.DataFrame(index=rng)
    base = 200 + 20*np.sin(2*np.pi*df.index.hour/24)
    weekday = df.index.weekday
    week_factor = np.where(weekday<5, 1.0, 0.85)
    day_of_year = df.index.dayofyear
    seasonal = 10*np.sin(2*np.pi*day_of_year/365)
    temp = 10 + 8*np.sin(2*np.pi*day_of_year/365) + 4*np.sin(2*np.pi*df.index.hour/24 + 1.0) + np.random.normal(0,1,size=hours)
    demand = (base * week_factor) + seasonal - 0.8*(temp-15) + np.random.normal(0,5,size=hours)
    df['demand'] = demand
    df['temp_C'] = temp
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['dayofyear'] = df.index.dayofyear
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data/energy.csv')
    parser.add_argument('--days', type=int, default=365)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    hours = args.days * 24
    df = generate(hours=hours, seed=args.seed)
    outdir = os.path.dirname(args.out)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir)
    df.to_csv(args.out, index_label='timestamp')
    print(f"Wrote {args.out} with {len(df)} rows.")

if __name__ == '__main__':
    main()
