import os
import time
from parser.markdown_parser import parse_markdown_chunks
from splitters.chunk_splitter import split_chunks
from vectorstore.embedding import embed_and_store_with_chroma
from vectorstore.chroma_utils import get_persist_dir
from eval.utils import find_duplicate_ids, find_duplicate_chunks
from config import embedding_models
from variables import table_markdown_file_name

def main():
    print("🔄 Parsing and chunking markdown...")
    chunks = parse_markdown_chunks(table_markdown_file_name)
    chunks = split_chunks(chunks, max_tokens=3000, overlap=333)
    find_duplicate_ids(chunks)
    find_duplicate_chunks(chunks)
    print(f"✅ Parsed and split into {len(chunks)} chunks.")

    for model in embedding_models:
        print(f"\n🧠 Embedding with model: {model}")
        persist_dir = get_persist_dir(model)

        # 👇 Create model-specific dir if it doesn’t exist
        os.makedirs(persist_dir, exist_ok=True)

        embed_and_store_with_chroma(chunks, model, persist_dir)
        print(f"✅ Saved to: {persist_dir}")
        

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"⏱️ Total time: {time.time() - start_time:.2f} seconds")
