from fastapi import FastAPI, Header, HTTPException, Request
from schema import QueryRequest, QueryResponse
from document_loader import extract_text_from_pdf
from embedder import chunk_text, embed_chunks, search_faiss
import tempfile
import requests
import os
import json
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            {"role": "system",
             "content": "You are a helpful assistant that answers questions based on clauses from legal, insurance, or compliance documents."},
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
async def run_query(request: QueryRequest, authorization: str = Header(...)):
    logger.info(f"Received request: {request}")
    logger.info(f"Authorization header: {authorization}")

    # Optional: validate token format (can be removed if not needed)
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization token")

    logger.info(f"Downloading PDF from: {request.documents}")

    try:
        response = requests.get(request.documents)
        response.raise_for_status()
        logger.info(f"PDF downloaded successfully, size: {len(response.content)} bytes")
    except requests.RequestException as e:
        logger.error(f"Failed to download PDF: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {str(e)}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    try:
        logger.info("Extracting text from PDF")
        full_text = extract_text_from_pdf(tmp_path)
        logger.info(f"Extracted text length: {len(full_text)} characters")

        logger.info("Chunking text")
        chunks = chunk_text(full_text)
        logger.info(f"Created {len(chunks)} chunks")

        logger.info("Creating embeddings")
        vectorstore = embed_chunks(chunks)

        answers = []
        for i, question in enumerate(request.questions):
            logger.info(f"Processing question {i + 1}/{len(request.questions)}: {question}")
            top_clauses = search_faiss(vectorstore, question)
            top_texts = [doc.page_content for doc in top_clauses]
            answer = generate_answer_with_groq(question, top_texts)
            answers.append(answer)
            logger.info(f"Generated answer for question {i + 1}")

        logger.info(f"Returning {len(answers)} answers")
        return QueryResponse(answers=answers)

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
            logger.info("Cleaned up temporary file")


# Add a simple test endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# Add endpoint to test JSON parsing
@app.post("/test")
async def test_endpoint(request: QueryRequest):
    return {"message": "JSON parsed successfully", "documents": request.documents,
            "question_count": len(request.questions)}