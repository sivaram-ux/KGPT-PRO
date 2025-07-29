def get_persist_dir(model_name):
    safe_name = model_name.replace("/", "_")
    return f"databases/chroma_db/{safe_name}"
