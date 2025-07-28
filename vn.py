from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


embedding = HuggingFaceEmbeddings(model_name="thenlper/gte-base")
db = Chroma.from_texts("texts", embedding=embedding, persist_directory="./chroma_db")
