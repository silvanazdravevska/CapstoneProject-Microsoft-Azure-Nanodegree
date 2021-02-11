import requests
import json

# URL for the web service, should be similar to:
# 'http://8530a665-66f3-49c8-a953-b82a2d312917.eastus.azurecontainer.io/score'
scoring_uri = ''

# Two sets of data to score, so we get two results back
data = {"data":
        [
          {
            "age": 55,
            "anaemia": 0,
            "creatinine_phosphokinase": 748,
            "diabetes": 0,
            "ejection_fraction": 35,
            "high_blood_pressure": 0,
            "platelets": 324000,
            "serum_creatinine": 0.7,
            "serum_sodium": 140,
            "sex": 1,
            "smoking": 0,
            "time": "88"
          },
          {
            "age": 66,
            "anaemia": 1,
            "creatinine_phosphokinase": 68,
            "diabetes": 1,
            "ejection_fraction": 38,
            "high_blood_pressure": 1,
            "platelets": 162000,
            "serum_creatinine": 1,
            "serum_sodium": 136,
            "sex": 0,
            "smoking": 0,
            "time": "95"
          },
      ]
    }
# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())


