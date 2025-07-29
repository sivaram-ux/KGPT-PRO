import os
import requests
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

def load_embed_model():
    return HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")

class InferenceEmbeddingWrapper:
    def __init__(self, model_id: str, token: str):
        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
        self.headers = {"Authorization": f"Bearer {token}"}

    def embed_query(self, text: str):
        try:
            res = requests.post(self.api_url, headers=self.headers, json={"inputs": text})
            res.raise_for_status()
            result = res.json()

            # Debug: show actual shape
            if isinstance(result[0], list):
                print(f"✅ Embedding dimension: {len(result[0])}")
                return result[0]
            else:
                print(f"✅ Embedding dimension: {len(result)}")
                return result

        except Exception as e:
            print("❌ Inference API embedding failed:", e)
            return [0.0] * 768  # ✅ Correct fallback dimension

def load_inference_wrapper():
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("⚠️ HF_TOKEN is not set")

    model_id = "intfloat/e5-base-v2"  # ✅ 768-dim, public, matches Chroma DB
    print("🔄 Using Hugging Face inference API for query embedding...")
    return InferenceEmbeddingWrapper(model_id, hf_token)
