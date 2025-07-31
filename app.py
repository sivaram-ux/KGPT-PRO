#start by using --> uvicorn app:app --reload --host 0.0.0.0 --port 8000

import os
from typing import AsyncGenerator
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

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

embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
persist_dir = get_persist_dir("models/embedding-001")
db = Chroma(persist_directory=persist_dir, embedding_function=embedding)
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.5})

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_api_key:
    raise ValueError("⚠️ OPENROUTER_API_KEY not found in .env file")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # Or "gemini-1.5-pro"
    temperature=0.7,
    streaming=True # Streaming is handled by the .stream() or .astream() methods
)


# ------------------------
# ✅ Helper Functions
# ------------------------
def retrieve_chunks(query: str, k: int = 5):
    """Retrieves document chunks synchronously."""
    docs = retriever.invoke(query)
    return [doc.page_content for doc in docs]

def build_prompt(query: str, chunks: list[str]) -> str:
    """Builds the prompt for the LLM."""
    return f"""As an expert assistant, your task is to provide a direct and clear answer to the user's question. Base your answer **exclusively** on the information available in the 'Retrieved Context' provided below. Do not mention or allude to the context in your response. Answer as if you know the information innately. Use emojis where appropriate."

Question:
{query}

Retrieved Context:
{chr(10).join(f"- {chunk}" for chunk in chunks)}
"""

async def stream_llm_response(prompt: str) -> AsyncGenerator[str, None]:
    """
    Yields response chunks from the LLM as they are generated using the async stream method.
    """
    try:
        # Use astream for async iteration
        async for chunk in llm.astream([HumanMessage(content=prompt)]):
            # The actual content is in the 'content' attribute of the chunk
            if content := chunk.content:
                yield content
    except Exception as e:
        print(f"LLM streaming error: {e}")
        # Optionally, you could yield an error message here, but it's often better
        # to let the connection close or handle it client-side.

# ------------------------
# ✅ RAG Endpoint (Streaming)
# ------------------------
@app.post("/query")
async def query_kgpt(data: QueryRequest):
    """
    Handles the user query, retrieves context, and streams the LLM response.
    """
    query = data.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # 1. Retrieve context chunks (this part is still synchronous)
        chunks = retrieve_chunks(query)
        
        # 2. Build the prompt
        prompt = build_prompt(query, chunks)
        
        # 3. Create the generator and return a streaming response
        response_generator = stream_llm_response(prompt)
        return StreamingResponse(response_generator, media_type="text/plain")

    except Exception as e:
        # This will catch errors from the synchronous parts (e.g., chunk retrieval)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")