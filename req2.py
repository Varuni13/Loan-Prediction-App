import requests

url = 'http://127.0.0.1:5000/predict'
payload = {
    "features": [2, 500, 100, 120, 50, 30, 20, 150, 10, 1, 0]
}
response = requests.post(url, json=payload)
print(response.json())
