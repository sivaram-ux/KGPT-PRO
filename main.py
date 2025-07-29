from langchain_chroma import Chroma
from vectorstore.chroma_utils import get_persist_dir
from config import default_model
from gemini.validator import ask_gemini_completion  # 👈 You will define this
import sys
from langchain_huggingface import HuggingFaceEmbeddings
from variables import eval_set
embedding = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")
persist_dir = get_persist_dir("intfloat/e5-base-v2")
db = Chroma(persist_directory=persist_dir,embedding_function=embedding)
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.5})
def retrieve_chunks(question, k=5):
    docs = retriever.invoke(question)
    return [doc.page_content for doc in docs]

def build_prompt(question, chunks):
    return f"""Answer the following question based **only** on the retrieved content below.

Question:
{question}

Retrieved Context:
{chr(10).join(f"- {chunk}" for chunk in chunks)}
"""

def main():
    for i,casee in enumerate(eval_set):
        

        print(f"\n🔍 Question {i+1}: {casee["question"]}")

        chunks = retrieve_chunks(casee["question"])
        print(f"\n📚 Retrieved {len(chunks)} chunks.\n")

        prompt = build_prompt(casee["question"], chunks)
        answer = ask_gemini_completion(prompt)
        print("\n🧠 Gemini Answer:\n")
        print(answer)

if __name__ == "__main__":
    main()
