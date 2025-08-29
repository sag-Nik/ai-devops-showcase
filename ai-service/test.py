import requests

url = "http://127.0.0.1:8000/analyze"
data = {"subreddit": "Iphone", "top_n": 25}

resp = requests.post(url, json=data)
print(resp.json())
