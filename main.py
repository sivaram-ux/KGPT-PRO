from parser.markdown_parser import parse_markdown_chunks
from splitters.chunk_splitter import split_chunks
from vectorstore.embedding import embed_and_store_with_chroma
from vectorstore.chroma_utils import get_persist_dir
from eval.semantic_eval import evaluate_retrieval_semantic
from eval.utils import find_duplicate_ids, find_duplicate_chunks
from config import embedding_models, default_model
from variables import table_markdown_file_name, eval_set

if __name__ == "__main__":
    chunks = parse_markdown_chunks(table_markdown_file_name)
    chunks = split_chunks(chunks, max_tokens=3000, overlap=333)
    find_duplicate_ids(chunks)
    find_duplicate_chunks(chunks)
    print(f"✅ Parsed and split into {len(chunks)} chunks.")

    for model in embedding_models:
        print(f"\n\n🔍 Using embedding model: {model}")
        db = embed_and_store_with_chroma(chunks, model, persist_directory=get_persist_dir(model))
        evaluate_retrieval_semantic(db, eval_set, model)
        print("\n" + "=" * 50 + "\n")
