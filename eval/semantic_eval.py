from sentence_transformers import SentenceTransformer, util
import numpy as np
from .utils import is_semantic_match, token_iou
from gemini.validator import ask_gemini_validation

def evaluate_retrieval_semantic(db, eval_set, model_name, k=5):
    semantic_model = SentenceTransformer(model_name)
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": k, "lambda_mult": 0.5})

    recall, mrr, ious = 0, 0, []
    gemini_yes, gemini_no = 0, 0
    failed_cases = []

    for idx, item in enumerate(eval_set):
        question, answer = item["question"], item["ground_truth"]
        docs = retriever.invoke(question)
        set_docs = set((doc.page_content, doc.metadata.get("id", "unknown")) for doc in docs)

        if len(docs) != len(set_docs):
            print(f"❗️ Duplicate documents found: {question}")

        found = False
        for rank, doc in enumerate(docs):
            if is_semantic_match(answer, doc.page_content, semantic_model):
                recall += 1
                mrr += 1 / (rank + 1)
                found = True
                break

        gemini_result = ask_gemini_validation(question, answer, [doc.page_content for doc in docs])
        if gemini_result.lower() == "yes": gemini_yes += 1
        elif gemini_result.lower() == "no": gemini_no += 1
        print(f"Question {idx+1} - Gemini says: {gemini_result}")

        best_iou = max(token_iou(answer, doc.page_content) for doc in docs)
        ious.append(best_iou)

        if not found:
            failed_cases.append({
                "index": idx,
                "question": question,
                "ground_truth": answer,
                "top_k_docs": [doc.page_content for doc in docs],
                "doc_names": [doc.metadata.get("id", "unknown") for doc in docs],
                "max_iou": best_iou
            })

    print(f"✅ Gemini Yes: {gemini_yes}, No: {gemini_no}, Total: {gemini_yes + gemini_no}")
    return failed_cases
