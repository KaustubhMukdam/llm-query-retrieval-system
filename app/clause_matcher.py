import requests
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or "your-groq-api-key-here" 

def generate_answer(query: str, relevant_clauses: list[str]) -> str:
    """
    Uses LLM (Groq) to answer the user's query based on the matched clauses.
    """
    context = "\n\n".join(relevant_clauses)[:8000]  

    prompt = f"""You are a legal assistant answering questions from policy documents.

Context:
{context}

Question:
{query}

Answer:"""

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 512
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"[Error generating response: {e}]"
