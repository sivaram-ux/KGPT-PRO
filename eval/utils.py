def is_semantic_match(answer, doc_text, model, threshold=0.75):
    emb1 = model.encode(answer, convert_to_tensor=True)
    emb2 = model.encode(doc_text, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2).item() >= threshold

def token_iou(a, b):
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    return len(set_a & set_b) / len(set_a | set_b) if set_a | set_b else 0

from collections import defaultdict, Counter
def find_duplicate_ids(chunks):
    id_map = defaultdict(list)
    for chunk in chunks:
        id_map[chunk["id"]].append(chunk["content"])
    for chunk_id, texts in id_map.items():
        if len(texts) > 1:
            print(f"❌ DUPLICATE ID FOUND: {chunk_id}")

def find_duplicate_chunks(chunks):
    contents = [chunk["content"] for chunk in chunks]
    dupes = [item for item, count in Counter(contents).items() if count > 1]
    print(f"❗️ Duplicate chunks: {len(dupes)}")
