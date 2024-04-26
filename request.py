import requests
url = 'http://localhost:5000/model'
r = requests.post(url,json={"input":["I love you"]})
print(r.json())