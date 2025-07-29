import os
import time
from dotenv import load_dotenv
from langchain_chroma import Chroma
from vectorstore.chroma_utils import get_persist_dir
from config import default_model
from gemini.validator import ask_gemini_completion
from load import load_embed_model, load_inference_wrapper
from variables import eval_set

# Track total script time
start_time = time.time()
print(f"⏱️ Script started at: {time.ctime(start_time)}")

# Step 1: Initialize Chroma with document DB and remote query embedder
print("\n🔄 Initializing embedding model and ChromaDB...")
step_start_time = time.time()

embedding = load_inference_wrapper()  # ⬅️ Using remote query embedder here
persist_dir = get_persist_dir("intfloat/e5-base-v2")
db = Chroma(persist_directory=persist_dir, embedding_function=embedding)
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.5})

print(f"✅ Embedding model and ChromaDB initialized in: {time.time() - step_start_time:.2f} seconds")

# Step 2: Retrieve chunks
query = "Tell me about TSG"

def retrieve_chunks(query, k=5):
    print(f"\n🔄 Retrieving chunks for query: '{query}'...")
    retrieval_start_time = time.time()
    docs = retriever.invoke(query)
    print(f"✅ Chunks retrieved in: {time.time() - retrieval_start_time:.2f} seconds")
    return [doc.page_content for doc in docs]

chunks = retrieve_chunks(query)

# Step 3: Build prompt
def build_prompt(query, chunks):
    print(f"\n🔄 Building prompt for LLM..., Length of chunks:{len(chunks)}")
    start = time.time()
    prompt_content = f"""Answer the following question based **only** on the retrieved content below.
    donot say any thing like "Based on the provided text" or "The text states that" or "According to the text" or "The text mentions that" or "The text says that" or "The text explains that" or "The text describes that" or "The text indicates that" or "The text reveals that" or "The text shows that" or "The text highlights that" or "The text suggests that" or "The text implies that".
    Use emojis if needed. Explain clearly.
Question:
{query}

Retrieved Context:
{chr(10).join(f"- {chunk}" for chunk in chunks)}
"""
    print(f"✅ Prompt built in: {time.time() - start:.2f} seconds")
    return prompt_content

prompt = build_prompt(query, chunks)

# Step 4: Load OpenRouter LLM
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY not set")

openrouter_base_url = "https://openrouter.ai/api/v1"
model_name = "google/gemma-3n-e4b-it:free"

print(f"\n🔄 Initializing ChatOpenAI with model: {model_name}...")
llm_init_start_time = time.time()
llm = ChatOpenAI(
    model=model_name,
    openai_api_base=openrouter_base_url,
    openai_api_key=openrouter_api_key,
    temperature=0.7
)
print(f"✅ ChatOpenAI initialized in: {time.time() - llm_init_start_time:.2f} seconds")

# Step 5: Run LLM on prompt
print("\n🔄 Invoking LLM for completion...")
llm_invoke_start_time = time.time()
try:
    response = llm.invoke([HumanMessage(content=prompt)])
    print("LLM Response:")
    print(response.content)
    print(f"✅ LLM invocation completed in: {time.time() - llm_invoke_start_time:.2f} seconds")
except Exception as e:
    print(f"❌ LLM error: {e}")
    print("Check OPENROUTER_API_KEY and model availability.")

# End summary
print(f"\n⏱️ Total script execution time: {time.time() - start_time:.2f} seconds")
