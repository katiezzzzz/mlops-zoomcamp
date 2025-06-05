import requests
from deepdiff import DeepDiff # for deep comparison of dictionaries

ride = {
    "PULocationID": 10,
    "DOLocationID": 50,
    "trip_distance": 40
}

url = 'http://localhost:9696/predict'
actual_response = requests.post(url, json=ride).json()
expected_response = {'duration': 22.2}

# check to nearest 0.1
diff = DeepDiff(actual_response, expected_response, significant_digits=1)
print(f'diff={diff}')

# only check the format
assert 'type_changes' not in diff
assert 'values_changed' not in diff