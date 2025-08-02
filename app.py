# app.py
import os
import uuid
from typing import AsyncGenerator, List, Dict
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory

from vectorstore.chroma_utils import get_persist_dir

# -------------------
# ✅ FastAPI Setup
# -------------------
load_dotenv()

app = FastAPI(title="KGPT RAG Server")

# CORS – lock down in production!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session cookie → signed UUID
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET", "replace-with-secure-random-32chars"),
    session_cookie="kgpt_session",
    max_age=60 * 60 * 24 * 7,  # 1 week
)

# ------------------------
# ✅ Request Schema
# ------------------------
class QueryRequest(BaseModel):
    query: str

# ------------------------
# ✅ RAG + Memory Store
# ------------------------
# Holds one ConversationBufferWindowMemory per session ID
session_memories: Dict[str, ConversationBufferWindowMemory] = {}

# ------------------------
# ✅ Environment + Models
# ------------------------
# Embeddings & Vectorstore
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
persist_dir = get_persist_dir("models/embedding-001")
db = Chroma(persist_directory=persist_dir, embedding_function=embedding)
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.5})

# LLM
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_api_key:
    raise ValueError("⚠️ OPENROUTER_API_KEY not found in .env file")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    streaming=True,
)

# ------------------------
# ✅ Helper Functions
# ------------------------
def retrieve_chunks(query: str, k: int = 5) -> List[str]:
    docs = retriever.invoke(query)
    return [doc.page_content for doc in docs]

# ------------------------
# ✅ RAG + Window-Memory Endpoint
# ------------------------
@app.post("/query")
async def query_kgpt(request: Request, data: QueryRequest):
    query = data.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # 1️⃣ Ensure a session ID
    sid = request.session.get("id")
    if not sid:
        sid = str(uuid.uuid4())
        request.session["id"] = sid

    # 2️⃣ Get or create this session's WindowMemory (last 5 turns)
    if sid not in session_memories:
        session_memories[sid] = ConversationBufferWindowMemory(
            k=5,
            return_messages=True,
            memory_key="history"
        )
    memory = session_memories[sid]

    # 3️⃣ Load the last 5 turns
    mem_vars = memory.load_memory_variables({})
    history: List[HumanMessage | AIMessage] = mem_vars.get("history", [])

    # 4️⃣ Retrieve RAG context
    chunks = retrieve_chunks(query)

    # 5️⃣ Build the full message list
    system_prompt = (
        "As an expert assistant, your task is to provide a direct and clear answer "
        "to the user's question. Base your answer **exclusively** on the information "
        "available in the 'Retrieved Context' below. Do not mention the context. "
        "Answer as if you know the information innately. Use emojis where appropriate."
    )

    messages = [
        SystemMessage(content=system_prompt),
        *history,
        SystemMessage(content="Retrieved Context:\n" + "\n".join(f"- {c}" for c in chunks)),
        HumanMessage(content=query),
    ]

    # 6️⃣ Stream the LLM response and capture it
    async def stream_gen() -> AsyncGenerator[str, None]:
        full_response = ""
        try:
            async for chunk in llm.astream(messages):
                if chunk.content:
                    full_response += chunk.content
                    yield chunk.content
        except Exception as e:
            print(f"LLM streaming error: {e}")
        finally:
            # 7️⃣ Save this turn into memory
            memory.save_context(
                {"input": query},
                {"output": full_response}
            )

    return StreamingResponse(stream_gen(), media_type="text/plain")
