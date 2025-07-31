import requests
import json

# Test data
url = "http://127.0.0.1:8000/api/v1/hackrx/run"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer b13a7ea500b71764dfeda2ee1183e43df372ce8f97bc74bcdec077c5a904fb0b"
}

payload = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?"
    ]
}

print("Testing /health endpoint first...")
health_response = requests.get("http://127.0.0.1:8000/health")
print(f"Health check: {health_response.status_code} - {health_response.text}")

print("\nTesting /test endpoint...")
test_response = requests.post("http://127.0.0.1:8000/test", json=payload, headers={"Content-Type": "application/json"})
print(f"Test endpoint: {test_response.status_code} - {test_response.text}")

print("\nTesting main /hackrx/run endpoint...")
try:
    response = requests.post(url, json=payload, headers=headers)
    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    print(f"Response Text: {response.text}")

    if response.status_code == 200:
        print("Success!")
        result = response.json()
        print(f"Answers: {result}")
    else:
        print("Error occurred")

except Exception as e:
    print(f"Exception: {str(e)}")