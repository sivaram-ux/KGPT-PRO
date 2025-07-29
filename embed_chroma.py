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

    # --- Start of new code ---
    output_file = "mychunks.txt"
    print(f"📝 Saving chunks to {output_file}...")
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks):
                # Assuming each chunk is a dictionary with 'id' and 'content' keys.
                # Adjust 'chunk.get("id")' and 'chunk.get("content")' if your chunk structure is different.
                chunk_id = chunk.get("id", f"unknown_id_{i}")
                chunk_content = chunk.get("content", "No content available.")
                f.write(f"--- Chunk ID: {chunk_id} ---\n")
                f.write(chunk_content)
                f.write("\n\n") # Add extra newline for separation between chunks
        print(f"✅ Chunks saved to {output_file}.")
    except Exception as e:
        print(f"❌ Error saving chunks to {output_file}: {e}")
    # --- End of new code ---

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
