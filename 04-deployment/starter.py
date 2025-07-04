#!/usr/bin/env python
# coding: utf-8

import argparse
import pickle
import pandas as pd

def read_data(filename):
    categorical = ['PULocationID', 'DOLocationID']
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def main(year, month):
    categorical = ['PULocationID', 'DOLocationID']

    # Load the model and DictVectorizer
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    # Read the data
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    df = read_data(url)

    # Prepare features
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)

    # Predict
    y_pred = model.predict(X_val)

    # Print mean predicted duration
    print('Mean predicted duration:', y_pred.mean())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score NYC Taxi Trip Data')
    parser.add_argument('--year', type=int, required=True, help='Year of the data')
    parser.add_argument('--month', type=int, required=True, help='Month of the data')
    args = parser.parse_args()

    main(args.year, args.month)
