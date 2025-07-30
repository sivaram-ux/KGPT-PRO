import os
import requests
from dotenv import load_dotenv
from huggingface_hub.utils import build_hf_headers
load_dotenv()



class InferenceEmbeddingWrapper:
    def __init__(self, model_id: str):
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        self.headers = build_hf_headers()

    def embed_query(self, text: str):
        try:
            res = requests.post(self.api_url, headers=self.headers, json={"inputs": text})
            res.raise_for_status()
            result = res.json()

            # Debug: show actual shape
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    return result[0] # Handles nested list: [[...]]
                return result # Handles flat list: [...]
            else:
                print(f"❌ Unexpected API response format: {result}")
                return [0.0] * 768

        except Exception as e:
            print("❌ Inference API embedding failed:", e)
            return [0.0] * 768  # ✅ Correct fallback dimension

def load_inference_wrapper():
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("⚠️ HF_TOKEN is not set")

    model_id = "BAAI/bge-base-en-v1.5"  # ✅ 768-dim, public, matches Chroma DB
    print("🔄 Using Hugging Face inference API for query embedding...")
    return InferenceEmbeddingWrapper(model_id)