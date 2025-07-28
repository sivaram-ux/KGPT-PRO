import json
import markdown
import numpy as np
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from variables import table_markdown_file_name, eval_set
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer, util

# -----------------------------
# 1. Parse markdown into chunks
# -----------------------------
def parse_markdown_chunks(markdown_path):
    with open(markdown_path, "r", encoding="utf-8") as f:
        raw_md = f.read()

    html = markdown.markdown(raw_md, extensions=["tables"])
    soup = BeautifulSoup(html, "html.parser")

    chunks = []
    current_title = ""
    chunk_id = 0

    for elem in soup.find_all(["h1", "h2", "h3", "p", "ul", "table"]):
        if elem.name in ["h1", "h2", "h3"]:
            current_title = elem.text.strip()

        elif elem.name == "table":
            print(f"[Table] Found at chunk: {chunk_id}")
            rows = elem.find_all("tr")
            headers = [td.get_text(strip=True) for td in rows[0].find_all(["th", "td"])]
            sentences = []
            for row in rows[1:]:
                values = [td.get_text(strip=True) for td in row.find_all("td")]
                row_dict = dict(zip(headers, values))
                flat = ", ".join([f"{k} is {v}" for k, v in row_dict.items()])
                sentences.append(flat)
            content = f"{current_title}:\n" + " ".join(sentences)

            chunks.append({
                "id": f"chunk_{chunk_id}",
                "title": current_title,
                "content": content
            })

        elif elem.name == "ul":
            items = [li.get_text(strip=True) for li in elem.find_all("li")]
            content = f"{current_title}:\n" + ", ".join(items)
            chunks.append({
                "id": f"chunk_{chunk_id}",
                "title": current_title,
                "content": content
            })

        elif elem.name == "p":
            text = elem.get_text(strip=True)
            if text:
                content = f"{current_title}:\n{text}"
                chunks.append({
                    "id": f"chunk_{chunk_id}",
                    "title": current_title,
                    "content": content
                })

        chunk_id += 1

    return chunks

# -----------------------------
# 2. Split chunks
# -----------------------------
def split_chunks(chunks, max_tokens=512, overlap=64):
    splitter = RecursiveCharacterTextSplitter(chunk_size=max_tokens, chunk_overlap=overlap)
    final_chunks = []
    for chunk in chunks:
        splits = splitter.split_text(chunk["content"])
        for i, split in enumerate(splits):
            final_chunks.append({
                "id": f"{chunk['id']}_part_{i}",
                "title": chunk["title"],
                "content": split
            })
    return final_chunks

# -----------------------------
# 3. Embed and store using Chroma
# -----------------------------
def embed_and_store_with_chroma(chunks, persist_directory="./chroma_db"):
    texts = [chunk["content"] for chunk in chunks]
    metadatas = [{"id": chunk["id"], "title": chunk["title"]} for chunk in chunks]

    print(f"🔍 Loading embedding model for Chroma...")
    embedding = HuggingFaceEmbeddings(model_name="thenlper/gte-base")

    print(f"🧠 Embedding and saving to Chroma DB...")
    db = Chroma.from_texts(
        texts=texts,
        embedding=embedding,
        metadatas=metadatas,
        persist_directory=persist_directory
    )

    db.persist()
    print("✅ Embedding complete and saved in:", persist_directory)
    return db

# -----------------------------
# 4. Semantic Evaluation
# -----------------------------
semantic_model = SentenceTransformer("thenlper/gte-base")

def is_semantic_match(answer, doc_text, threshold=0.75):
    emb1 = semantic_model.encode(answer, convert_to_tensor=True)
    emb2 = semantic_model.encode(doc_text, convert_to_tensor=True)
    score = util.pytorch_cos_sim(emb1, emb2).item()
    return score >= threshold

def token_iou(a, b):
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0

def evaluate_retrieval_semantic(db, eval_set, k=2):
    retriever = db.as_retriever(search_kwargs={"k": k})
    recall_count = 0
    mrr_total = 0
    ious = []

    for item in eval_set:
        question = item["question"]
        answer = item["ground_truth"]
        docs = retriever.get_relevant_documents(question)

        found = False
        for rank, doc in enumerate(docs):
            if is_semantic_match(answer, doc.page_content):
                recall_count += 1
                mrr_total += 1 / (rank + 1)
                found = True
                break

        best_iou = max(token_iou(answer, doc.page_content) for doc in docs)
        ious.append(best_iou)

    print("📊 Semantic Evaluation Results:")
    print(f"   ✅ Recall@{k}: {recall_count / len(eval_set):.2f}")
    print(f"   ✅ MRR: {mrr_total / len(eval_set):.2f}")
    print(f"   ✅ Avg Token IoU: {np.mean(ious):.2f}")

# -----------------------------
# 5. Main
# -----------------------------
if __name__ == "__main__":
    chunks = parse_markdown_chunks(table_markdown_file_name)
    chunks = split_chunks(chunks, max_tokens=512, overlap=64)
    print(f"✅ Parsed and split into {len(chunks)} chunks.")

    db = embed_and_store_with_chroma(chunks)
    evaluate_retrieval_semantic(db, eval_set)
