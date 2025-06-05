import pickle
from flask import Flask, request
import json
import os

PATH = os.path.dirname(os.path.realpath(__file__))

def get_model_path():
    model_location = os.getenv('MODEL_LOCATION')
    if model_location is not None:
        return model_location
    else:
        return PATH

def load_model():
    model_location = get_model_path()
    with open(f'{model_location}/lin_reg.bin', 'rb') as f_in:
        (dv, model) = pickle.load(f_in)
    return dv, model

class ModelService():
    def __init__(self, dv, model):
        self.dv = dv
        self.model = model

    def prepare_features(self, ride):
        features = {}
        features['PU_DO'] = '%s_%s' % (ride["PULocationID"], ride["DOLocationID"])
        features['trip_distance'] = ride["trip_distance"]
        return features

    def predict(self, features):
        X = self.dv.transform(features)
        preds = self.model.predict(X)
        return float(preds[0])
    
    def predict_endpoint(self, ride):
        features = self.prepare_features(ride)
        pred = self.predict(features)
        result = {'duration': pred}
        return json.dumps(result)

app = Flask('duration-prediction')

def init():
    dv, model = load_model()
    model_service = ModelService(dv, model)
    return model_service

# turning function into http endpoint
@app.route('/predict', methods=['POST'])
def predict():
    model_service = init()
    ride = request.get_json()
    return model_service.predict_endpoint(ride)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
