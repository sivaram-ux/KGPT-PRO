from langchain_chroma import Chroma  # New official vector store
from langchain_huggingface import HuggingFaceEmbeddings

# 🔁 Load the exact same model you used for storage
embedding = HuggingFaceEmbeddings(model_name="thenlper/gte-base")

# ✅ Load Chroma DB AND attach embedding explicitly
retriever = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding  # ✅ required now
).as_retriever()

# 🔍 Query
query = "What is TSG at IIT Kharagpur?"
results = retriever.invoke(query)

# 🧾 Output
for i, res in enumerate(results):
    print(f"\nResult {i+1}:\n{res.page_content}")
    print("Metadata:", res.metadata)
