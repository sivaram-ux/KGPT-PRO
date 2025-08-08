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

# 1) CORS – lock to your deployed frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ← replace with your front-end URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2) Session cookie → signed UUID (SameSite=None, Secure for cross-site POST over HTTPS)
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET", "replace-with-secure-random-32chars"),
    session_cookie="kgpt_session",
    max_age=60 * 60 * 24 * 7,   # 1 week
    same_site="none",           # allow cross-site
    https_only=True             # only over HTTPS
)

# ------------------------
# ✅ Request Schema 
# ------------------------
class QueryRequest(BaseModel):
    query: str

# ------------------------
# ✅ RAG + Memory Store
# ------------------------
# one ConversationBufferWindowMemory per session ID
session_memories: Dict[str, ConversationBufferWindowMemory] = {}

# ------------------------
# ✅ Environment + Models
# ------------------------
# Embeddings & Vectorstore
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
persist_dir = get_persist_dir("models/embedding-001")
os.makedirs(persist_dir, exist_ok=True)
db = Chroma(persist_directory=persist_dir, embedding_function=embedding)
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.8})

# LLM
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_api_key:
    raise ValueError("⚠️ OPENROUTER_API_KEY not found in .env file")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    streaming=True,
)

llm_less_temp = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    streaming=True,
)

# ------------------------
# ✅ Helper Functions
# ------------------------
def retrieve_chunks(query: str, k: int = 9) -> List[str]:
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
    
    # 1️⃣ Ensure a session ID in the signed cookie
    sid = request.session.get("id")
    if not sid:
        sid = str(uuid.uuid4())
        request.session["id"] = sid

    # 2️⃣ Get or create this session's 5-turn window memory
    if sid not in session_memories:
        session_memories[sid] = ConversationBufferWindowMemory(
            k=9,
            return_messages=True,
            memory_key="history"
        )
    memory = session_memories[sid]

    # 3️⃣ Load the last 5 turns from memory
    mem_vars = memory.load_memory_variables({})
    history: List[HumanMessage | AIMessage] = mem_vars.get("history", [])

    system_prompt_for_updating_query =(f"give me the best version of query to use for RAG, first of check the history of the conversation, ")
    messages = [
        SystemMessage(content=(
    You are a query refinement assistant for a Retrieval-Augmented Generation (RAG) system.

Your task:
- Read the provided conversation history to understand the student's intent.
- Use the latest student query as the main focus.
- Expand it with related keywords or relevant phrases to improve retrieval results.
- Keep the technical meaning exactly the same.
- Do not expand abbreviations or full forms unless the user already spelled them out.
- Return only the refined query, without any explanation or extra text.

You must output a single query string.

)),
*history,
HumanMessage(content="Query: " + query)

    ]
    query = llm_less_temp.invoke(messages).content.strip()
    print(f"Refined query for RAG: {query}")


    # 4️⃣ Retrieve RAG context
    chunks = retrieve_chunks(query)

    # 5️⃣ Build the full message sequence
    system_prompt = (
        You are an expert assistant for answering questions based on IIT Kharagpur-related knowledge using a hidden Retrieval-Augmented Generation (RAG) process.

Your rules:
- The user must never know that you have retrieved any external context or documents.
- Never mention, refer to, or hint about having any "retrieved context", "sources", "documents", or "search results".
- Present your answer as if it comes entirely from your own internal knowledge as a highly trained LLM on IIT KGP data.
- Base your answer exclusively on the retrieved context provided internally to you.
- If there is insufficient information to answer, apologize politely and simply say you cannot answer — without mentioning context availability.
- Give direct, clear, and well-structured responses.
- Use emojis where appropriate to make answers more engaging.

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
            # 7️⃣ Save this turn into memory (auto-prunes oldest if >5)
            memory.save_context(
                {"input": query},
                {"output": full_response}
            )

    return StreamingResponse(stream_gen(), media_type="text/plain")
