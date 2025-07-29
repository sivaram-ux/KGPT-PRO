from langchain_community.vectorstores import Chroma
from config import embedding_models
from variables import eval_set
from vectorstore.chroma_utils import get_persist_dir
from eval.semantic_eval import evaluate_retrieval_semantic

def main():
    print("🔍 Running evaluations...\n")
    for model in embedding_models:
        persist_dir = get_persist_dir(model)
        print(f"📂 Loading Chroma DB from: {persist_dir}")
        db = Chroma(persist_directory=persist_dir)
        evaluate_retrieval_semantic(db, eval_set, model)
        print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
