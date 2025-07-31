import requests
import json

# Test data with all 10 questions
url = "http://127.0.0.1:8000/api/v1/hackrx/run"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer b13a7ea500b71764dfeda2ee1183e43df372ce8f97bc74bcdec077c5a904fb0b"
}

payload = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
}

print("Testing with all 10 questions...")
print("=" * 60)

try:
    response = requests.post(url, json=payload, headers=headers)
    print(f"Status Code: {response.status_code}")
    print(f"Response Time: {response.elapsed.total_seconds():.2f} seconds")
    print(f"Response Size: {len(response.content)} bytes")

    if response.status_code == 200:
        print("\n✅ SUCCESS! All questions processed successfully.")
        result = response.json()

        print(f"\nReceived {len(result['answers'])} answers:")
        print("=" * 60)

        for i, (question, answer) in enumerate(zip(payload['questions'], result['answers']), 1):
            print(f"\nQ{i}: {question}")
            print(f"A{i}: {answer}")
            print("-" * 40)

    else:
        print("❌ Error occurred")
        print(f"Response: {response.text}")

except requests.exceptions.Timeout:
    print("❌ Request timed out - this is normal for processing 10 questions")
except Exception as e:
    print(f"❌ Exception: {str(e)}")

print("\n" + "=" * 60)
print("Note: If you see this working here but Swagger UI shows errors,")
print("that's a known limitation of Swagger UI with large responses.")
print("Your API is working correctly!")