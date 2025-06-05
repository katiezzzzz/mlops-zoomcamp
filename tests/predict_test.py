from web_service import predict

def test_prepare_features():
    ride = {
        "PULocationID": 130,
        "DOLocationID": 205,
        "trip_distance": 3.66
    }
    actual_features = predict.prepare_features(ride)
    expected_features = {
        "PU_DO": "130_205",
        "trip_distance": 3.66
    }
    assert actual_features == expected_features