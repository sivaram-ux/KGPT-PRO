#start by using --> uvicorn app:app --reload --host 0.0.0.0 --port 8000

import os
import time
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from load import load_inference_wrapper
from vectorstore.chroma_utils import get_persist_dir
# -------------------
# ✅ FastAPI Setup
# -------------------
app = FastAPI(title="KGPT RAG Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# ✅ Request Schema
# ------------------------
class QueryRequest(BaseModel):
    query: str

# ------------------------
# ✅ Environment + Models
# ------------------------
load_dotenv()

embedding = load_inference_wrapper()
persist_dir = get_persist_dir("intfloat/e5-base-v2")
db = Chroma(persist_directory=persist_dir, embedding_function=embedding)
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.5})

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_api_key:
    raise ValueError("⚠️ OPENROUTER_API_KEY not found in .env file")

llm = ChatOpenAI(
    model="google/gemma-3n-e4b-it:free",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=openrouter_api_key,
    temperature=0.7
)

# ------------------------
# ✅ Helper Functions
# ------------------------
def retrieve_chunks(query: str, k: int = 5):
    docs = retriever.invoke(query)
    return [doc.page_content for doc in docs]

def build_prompt(query: str, chunks: list[str]) -> str:
    return f"""Answer the following question based **only** on the retrieved content below.
donot say any thing like "Based on the provided text" or "The text states that" or "According to the text" or "The text mentions that" or "The text says that" or "The text explains that" or "The text describes that" or "The text indicates that" or "The text reveals that" or "The text shows that" or "The text highlights that" or "The text suggests that" or "The text implies that".
Use emojis if needed. Explain clearly.

Question:
{query}

Retrieved Context:
{chr(10).join(f"- {chunk}" for chunk in chunks)}
"""

# ------------------------
# ✅ RAG Endpoint
# ------------------------
@app.post("/query")
def query_kgpt(data: QueryRequest):
    start_time = time.time()
    query = data.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        chunks = retrieve_chunks(query)
        prompt = build_prompt(query, chunks)
        response = llm.invoke([HumanMessage(content=prompt)])
        return {
            "query": query,
            "response": response.content,
            "time_taken": round(time.time() - start_time, 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")
