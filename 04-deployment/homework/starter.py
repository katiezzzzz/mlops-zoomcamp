import sys
import pickle
import pandas as pd

taxi_type = sys.argv[1]
year = int(sys.argv[2])
month = int(sys.argv[3])
categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = pd.to_datetime(df["tpep_dropoff_datetime"]) - pd.to_datetime(df["tpep_pickup_datetime"])
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def load_model():
    with open('./models/lin_reg.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model


def apply_model(input_file, output_file):
    print(f"reading data from {input_file}...")
    df = read_data(input_file)
    dicts = df[categorical].to_dict(orient='records')

    print("loading model...")
    dv, model = load_model()

    print("running model...")
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    print(f"saving results to {output_file}...")
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred
    print(f"mean predicted duration: {df_result['predicted_duration'].mean()}")
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

def run():
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'./output/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'

    apply_model(input_file, output_file)



if __name__ == '__main__':
    run()

