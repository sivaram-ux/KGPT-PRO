import requests
import time
start_time = time.time()
url = "https://lmjl8t57-8000.inc1.devtunnels.ms/embed"
headers = {
    "Content-Type": "application/json"
}
payload = {
    "text": "The future of AI is modular and open."
}

response = requests.post(url, headers=headers, json=payload)

if response.status_code == 200:
    embedding = response.json()["embedding"]
    print("✅ Embedding:", embedding[:5], "...")  # print first 5 values
else:
    print("❌ Error:", response.status_code, response.text)
    print(response.json())

print(f"⏱️ Total time: {time.time() - start_time:.2f} seconds")