from fastapi import FastAPI, Header, HTTPException
from app.schema import QueryRequest, QueryResponse
from app.document_loader import extract_text_from_pdf
from app.embedder import chunk_text, embed_chunks, search_faiss
import tempfile
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def generate_answer_with_groq(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)[:8000]
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on clauses from legal, insurance, or compliance documents."},
            {"role": "user", "content": f"Document context:\n{context}\n\nQuestion: {question}"}
        ],
        "temperature": 0.2,
        "max_tokens": 512
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"[Error from Groq API: {str(e)}]"

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
def run_query(request: QueryRequest, authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization token")

    response = requests.get(request.documents)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    full_text = extract_text_from_pdf(tmp_path)
    chunks = chunk_text(full_text)
    vectorstore = embed_chunks(chunks)

    answers = []
    for question in request.questions:
        top_clauses = search_faiss(vectorstore, question)
        top_texts = [doc.page_content for doc in top_clauses]
        answer = generate_answer_with_groq(question, top_texts)
        answers.append(answer)

    return QueryResponse(answers=answers)
