import shutil
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb.errors import InternalError
from .chroma_utils import get_persist_dir

def embed_and_store_with_chroma(chunks, model_name, persist_directory):
    texts = [chunk["content"] for chunk in chunks]
    ids = [chunk["id"] for chunk in chunks]
    metadatas = [{"id": chunk["id"]} for chunk in chunks]

    def build_chroma():
        return Chroma.from_texts(
            texts=texts,
            ids=ids,
            metadatas=metadatas,
            embedding=HuggingFaceEmbeddings(model_name=model_name),
            persist_directory=persist_directory
        )

    try:
        return build_chroma()
    except InternalError as e:
        if "readonly database" in str(e).lower():
            print("❗️ Read-only DB error. Resetting Chroma DB...")
            shutil.rmtree(persist_directory, ignore_errors=True)
            return build_chroma()
        else:
            raise
