import predict
import requests
import json
import pickle
import os

PATH = os.path.dirname(os.path.realpath(__file__))

ride = {
    "PULocationID": 10,
    "DOLocationID": 50,
    "trip_distance": 40
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=ride)
print(response.json())

# with open(f'{PATH}/lin_reg.bin', 'rb') as f_in:
#     (dv, model) = pickle.load(f_in)

# model_service = predict.ModelService(dv, model)
# features = model_service.prepare_features(ride)
# pred = model_service.predict(features)
# result = {'duration': pred}
# print(json.dumps(result))