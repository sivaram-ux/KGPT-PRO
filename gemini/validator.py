from dotenv import load_dotenv
load_dotenv()

from google.generativeai import GenerativeModel
from google.api_core.exceptions import GoogleAPICallError
import logging

# Initialize models from best to worst
model_names = [
    
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-2.5-pro"
]

# Cache the working model
working_model = None

def ask_gemini_validation(question, answer, retrieved_docs):
    global working_model

    prompt = f"""
You are validating a retrieval-based question answering system.

Question:
{question}

Expected Answer:
{answer}

Retrieved Chunks:
{retrieved_docs}

Based on the above chunks, can the question be accurately answered?
Respond with "Yes" or "No".No other explanation.
""".strip()

    # If we already have a working model, try it first
    if working_model:
        try:
            return working_model.generate_content(prompt).text.strip()
        except:
            working_model = None  # Invalidate and fallback to search

    # Try each model only until one works
    for name in model_names:
        try:
            model = GenerativeModel(name)
            result = model.generate_content(prompt)
            working_model = model  # Cache the successful one
            return result.text.strip()
        except:
            pass

    return "❌ All models failed. Try again later."
