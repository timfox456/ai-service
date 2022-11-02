import requests


req = { "url": "http://bit.ly/mlbookcamp-pants" }


url = 'http://localhost:9696/predict'
response = requests.post(url, json=req) 
response.json()
