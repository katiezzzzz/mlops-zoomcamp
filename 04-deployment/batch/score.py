from sklearn.feature_extraction import DictVectorizer
from mlflow.tracking import MlflowClient

import pandas as pd
import pickle
import mlflow
import uuid
import sys


def read_dataframe(filename: str):
    df = pd.read_csv(filename)

    df['duration'] = pd.to_datetime(df["lpep_dropoff_datetime"]) - pd.to_datetime(df["lpep_pickup_datetime"])
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    df['ride_ids'] = generate_uuids(len(df))
    return df


def prepare_data(df: pd.DataFrame, dv: DictVectorizer):
    duration = df["duration"].to_numpy()
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    X = dv.transform(dicts)
    # X = xgb.DMatrix(X, label=duration)
    return X


def load_model(run_id):
    logged_model = f'runs:/{run_id}/model'
    model = mlflow.pyfunc.load_model(logged_model)
    with open("models/preprocessor.b", "rb") as f_in:
        dv = pickle.load(f_in)
    return dv, model

def generate_uuids(n):
    ride_ids = []
    for i in range(n):
        ride_ids.append(str(uuid.uuid4()))
    return ride_ids

def apply_model(input_file, run_id, output_file):
    print(f'reading data from {input_file}...')
    df = read_dataframe(input_file)

    print(f'loading model with RUN_ID={run_id}...')
    dv, model = load_model(run_id)

    print('applying model...')
    X = prepare_data(df, dv)
    y_pred = model.predict(X)

    print(f'saving results to {output_file}...')
    df_result = pd.DataFrame()
    df_result['ride_ids'] = df['ride_ids']
    df_result['lpep_pickup_datetime'] = df['lpep_pickup_datetime']
    df_result['PULocationID'] = df['PULocationID']
    df_result['DOLocationID'] = df['DOLocationID']
    df_result['actual_duration'] = df['duration']
    df_result['predicted_duration'] = y_pred
    df_result['diff'] = df_result['actual_duration'] - df_result['predicted_duration']
    df_result['model_version'] = run_id

    df_result.to_csv(output_file, index=False)

def run():
    taxi_type = sys.argv[1] # 'green'
    year = int(sys.argv[2]) # 2024
    month = int(sys.argv[3]) # 1

    input_file = f'./data/{taxi_type}_tripdata_{year:04d}-{month:02d}.csv'
    output_file = f'./output/{taxi_type}_tripdata_{year:04d}-{month:02d}.csv'

    RUN_ID = 'e73c0f6b3b754e80bcfb3be72c0c2f40'
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("green-taxi-duration")

    apply_model(input_file, RUN_ID, output_file)



if __name__ == '__main__':
    run()
