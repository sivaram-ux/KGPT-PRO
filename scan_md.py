import json
from langchain_text_splitters import TokenTextSplitter
import markdown
from bs4 import BeautifulSoup
from variables import *
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

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
def split_chunks(chunks, max_tokens=500, overlap=50):
    splitter = TokenTextSplitter(chunk_size=max_tokens, chunk_overlap=overlap)
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

    print(f"🔍 Loading embedding model...")
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
# 4. Main runner
# -----------------------------
if __name__ == "__main__":
    chunks = parse_markdown_chunks(table_markdown_file_name)
    chunks = split_chunks(chunks, max_tokens=500, overlap=50)
    print(f"Parsed {len(chunks)} chunks from {table_markdown_file_name}.\n")

    for i in range(max(5, len(chunks))):
        print(i,"  ",len(chunks[i]["content"]), "\n\n")

    #embed_and_store_with_chroma(chunks)
