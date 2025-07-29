from langchain_text_splitters import RecursiveCharacterTextSplitter

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
