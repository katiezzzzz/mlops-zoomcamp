from web_service import predict

def test_prepare_features():
    model_service = predict.ModelService(None, None)
    ride = {
        "PULocationID": 130,
        "DOLocationID": 205,
        "trip_distance": 3.66
    }
    actual_features = model_service.prepare_features(ride)
    expected_features = {
        "PU_DO": "130_205",
        "trip_distance": 3.66
    }
    assert actual_features == expected_features

# Mock class to simulate the model, without actually loading it
class ModelMock:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        n = len(X)
        return [10.0] * n
    
class DictVectorizerMock:
    def transform(self, features):
        # Simulate transformation by returning a dummy array
        return [[1] * len(features)]  # Dummy transformation
    
def test_predict():
    model = ModelMock(10)
    dv = DictVectorizerMock()
    model_service = predict.ModelService(dv, model)

    features = {
        "PU_DO": "130_205",
        "trip_distance": 3.66
    }

    actual_prediction = model_service.predict(features)
    expected_prediction = 10

    assert actual_prediction == expected_prediction