# from fastapi import FastAPI, Header, HTTPException, Request
# from schema import QueryRequest, QueryResponse
# from document_loader import extract_text_from_pdf
# from embedder import chunk_text, embed_chunks, search_faiss
# import tempfile
# import requests
# import os
# import json
# import logging
# from dotenv import load_dotenv

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# load_dotenv()

# app = FastAPI()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# def generate_answer_with_groq(question: str, context_chunks: list[str]) -> str:
#     context = "\n\n".join(context_chunks)[:8000]
#     url = "https://api.groq.com/openai/v1/chat/completions"
#     headers = {
#         "Authorization": f"Bearer {GROQ_API_KEY}",
#         "Content-Type": "application/json"
#     }
#     payload = {
#         "model": "llama3-70b-8192",
#         "messages": [
#             {"role": "system",
#              "content": "You are a helpful assistant that answers questions based on clauses from legal, insurance, or compliance documents."},
#             {"role": "user", "content": f"Document context:\n{context}\n\nQuestion: {question}"}
#         ],
#         "temperature": 0.2,
#         "max_tokens": 512
#     }
#     try:
#         response = requests.post(url, headers=headers, json=payload)
#         response.raise_for_status()
#         return response.json()['choices'][0]['message']['content'].strip()
#     except Exception as e:
#         return f"[Error from Groq API: {str(e)}]"


# @app.post("/api/v1/hackrx/run", response_model=QueryResponse)
# async def run_query(request: QueryRequest, authorization: str = Header(...)):
#     logger.info(f"Received request: {request}")
#     logger.info(f"Authorization header: {authorization}")

#     # Optional: validate token format (can be removed if not needed)
#     if not authorization.startswith("Bearer "):
#         raise HTTPException(status_code=401, detail="Invalid Authorization token")

#     logger.info(f"Downloading PDF from: {request.documents}")

#     try:
#         response = requests.get(request.documents)
#         response.raise_for_status()
#         logger.info(f"PDF downloaded successfully, size: {len(response.content)} bytes")
#     except requests.RequestException as e:
#         logger.error(f"Failed to download PDF: {str(e)}")
#         raise HTTPException(status_code=400, detail=f"Failed to download PDF: {str(e)}")

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#         tmp.write(response.content)
#         tmp_path = tmp.name

#     try:
#         logger.info("Extracting text from PDF")
#         full_text = extract_text_from_pdf(tmp_path)
#         logger.info(f"Extracted text length: {len(full_text)} characters")

#         logger.info("Chunking text")
#         chunks = chunk_text(full_text)
#         logger.info(f"Created {len(chunks)} chunks")

#         logger.info("Creating embeddings")
#         vectorstore = embed_chunks(chunks)

#         answers = []
#         for i, question in enumerate(request.questions):
#             logger.info(f"Processing question {i + 1}/{len(request.questions)}: {question}")
#             top_clauses = search_faiss(vectorstore, question)
#             top_texts = [doc.page_content for doc in top_clauses]
#             answer = generate_answer_with_groq(question, top_texts)
#             answers.append(answer)
#             logger.info(f"Generated answer for question {i + 1}")

#         logger.info(f"Returning {len(answers)} answers")
#         return QueryResponse(answers=answers)

#     except Exception as e:
#         logger.error(f"Error processing request: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
#     finally:
#         # Clean up temporary file
#         if os.path.exists(tmp_path):
#             os.unlink(tmp_path)
#             logger.info("Cleaned up temporary file")


# # Add a simple test endpoint
# @app.get("/health")
# async def health_check():
#     return {"status": "healthy"}


# # Add endpoint to test JSON parsing
# @app.post("/test")
# async def test_endpoint(request: QueryRequest):
#     return {"message": "JSON parsed successfully", "documents": request.documents,
#             "question_count": len(request.questions)}

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from schema import QueryRequest, QueryResponse
from document_loader import extract_text_from_pdf
from embedder import chunk_text, embed_chunks, search_faiss
import tempfile
import requests
import os
import json
import logging
import time
from dotenv import load_dotenv
from typing import List
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(
    title="LLM Query Retrieval System",
    description="Intelligent document query system for insurance, legal, HR, and compliance domains",
    version="1.0.0"
)

# Add CORS middleware for web testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=5)

def download_document_with_retry(url: str, max_retries: int = 3) -> bytes:
    """Download document with retry logic and better error handling."""
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting to download document (attempt {attempt + 1}/{max_retries})")
            
            # Set timeout and headers for better compatibility
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/pdf,application/octet-stream,*/*'
            }
            
            response = requests.get(
                url, 
                headers=headers,
                timeout=60,  # 60 second timeout
                stream=True,
                verify=True
            )
            response.raise_for_status()
            
            # Validate content type
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type and 'octet-stream' not in content_type:
                logger.warning(f"Unexpected content type: {content_type}")
            
            content = response.content
            if len(content) == 0:
                raise ValueError("Downloaded file is empty")
                
            logger.info(f"Successfully downloaded document: {len(content)} bytes")
            return content
            
        except requests.exceptions.Timeout:
            logger.warning(f"Download timeout on attempt {attempt + 1}")
            if attempt == max_retries - 1:
                raise HTTPException(status_code=408, detail="Document download timeout")
            time.sleep(2 ** attempt)  # Exponential backoff
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Download error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")
            time.sleep(2 ** attempt)
            
        except Exception as e:
            logger.error(f"Unexpected error downloading document: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Document download failed: {str(e)}")


def generate_answer_with_groq(question: str, context_chunks: List[str], max_retries: int = 3) -> str:
    """Generate answer using Groq API with retry logic and better prompting."""
    
    if not GROQ_API_KEY or GROQ_API_KEY == "your-groq-api-key-here":
        return "[Error: Groq API key not configured]"
    
    # Combine and truncate context
    context = "\n\n".join(context_chunks)
    if len(context) > 8000:
        context = context[:8000] + "...[truncated]"
    
    # Enhanced prompt for better accuracy
    prompt = f"""You are an expert assistant specializing in analyzing insurance, legal, HR, and compliance documents. 

Based on the provided document context, answer the question accurately and concisely.

DOCUMENT CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Provide a clear, specific answer based only on the information in the context
- If the context doesn't contain enough information, state what information is missing
- Include relevant details like timeframes, conditions, or limitations when applicable
- Keep the answer concise but complete

ANSWER:"""

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant that provides accurate answers based on document context. You specialize in insurance, legal, HR, and compliance domains."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,  # Lower temperature for more consistent answers
        "max_tokens": 512,
        "top_p": 0.9
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url, 
                headers=headers, 
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            answer = result['choices'][0]['message']['content'].strip()
            
            if not answer:
                raise ValueError("Empty response from Groq API")
                
            return answer
            
        except requests.exceptions.Timeout:
            logger.warning(f"Groq API timeout on attempt {attempt + 1}")
            if attempt == max_retries - 1:
                return f"[Error: API timeout after {max_retries} attempts]"
            time.sleep(1)
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Groq API request error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                return f"[Error: API request failed - {str(e)}]"
            time.sleep(1)
            
        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"Error parsing Groq API response: {str(e)}")
            return f"[Error: Invalid API response - {str(e)}]"
            
        except Exception as e:
            logger.error(f"Unexpected error calling Groq API: {str(e)}")
            return f"[Error: Unexpected API error - {str(e)}]"


def process_single_question(vectorstore, question: str, question_num: int) -> str:
    """Process a single question - for potential parallel processing."""
    try:
        logger.info(f"Processing question {question_num}: {question[:50]}...")
        
        # Retrieve relevant chunks
        top_clauses = search_faiss(vectorstore, question, k=5)
        
        if not top_clauses:
            return "[Error: No relevant content found in document]"
        
        top_texts = [doc.page_content for doc in top_clauses if hasattr(doc, 'page_content')]
        
        if not top_texts:
            return "[Error: Unable to extract relevant text chunks]"
        
        # Generate answer
        answer = generate_answer_with_groq(question, top_texts)
        logger.info(f"Completed question {question_num}")
        
        return answer
        
    except Exception as e:
        logger.error(f"Error processing question {question_num}: {str(e)}")
        return f"[Error processing question: {str(e)}]"


@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def run_query(request: QueryRequest, authorization: str = Header(...)):
    """
    Main endpoint for processing document queries.
    Handles PDF documents and returns structured JSON responses.
    """
    start_time = time.time()
    logger.info(f"Received request with {len(request.questions)} questions")
    logger.info(f"Document URL: {request.documents}")

    # Validate input
    if not request.questions:
        raise HTTPException(status_code=400, detail="No questions provided")
    
    if len(request.questions) > 50:  # Reasonable limit
        raise HTTPException(status_code=400, detail="Too many questions (max 50)")

    # Validate authorization header format
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")

    tmp_path = None
    try:
        # Download document
        logger.info("Downloading document...")
        document_content = download_document_with_retry(request.documents)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(document_content)
            tmp_path = tmp.name

        # Extract text
        logger.info("Extracting text from document...")
        full_text = extract_text_from_pdf(tmp_path)
        
        if not full_text or len(full_text.strip()) < 100:
            raise HTTPException(status_code=400, detail="Document appears to be empty or unreadable")
        
        logger.info(f"Extracted {len(full_text)} characters from document")

        # Create chunks
        logger.info("Creating text chunks...")
        chunks = chunk_text(full_text)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Failed to create text chunks from document")
        
        logger.info(f"Created {len(chunks)} text chunks")

        # Create embeddings and vector store
        logger.info("Building vector store...")
        vectorstore = embed_chunks(chunks)

        # Process questions
        logger.info(f"Processing {len(request.questions)} questions...")
        answers = []
        
        # Process questions sequentially for stability
        # (Can be parallelized if needed, but sequential is more reliable)
        for i, question in enumerate(request.questions, 1):
            answer = process_single_question(vectorstore, question, i)
            answers.append(answer)

        processing_time = time.time() - start_time
        logger.info(f"Successfully processed all questions in {processing_time:.2f} seconds")

        return QueryResponse(answers=answers)

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
    finally:
        # Clean up temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                logger.info("Cleaned up temporary file")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "LLM Query Retrieval System",
        "version": "1.0.0",
        "groq_api_configured": bool(GROQ_API_KEY and GROQ_API_KEY != "your-groq-api-key-here")
    }


@app.post("/test")
async def test_endpoint(request: QueryRequest):
    """Test endpoint for validating JSON parsing."""
    return {
        "message": "JSON parsed successfully",
        "documents": request.documents,
        "question_count": len(request.questions),
        "questions_preview": request.questions[:3] if len(request.questions) > 3 else request.questions
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "LLM Query Retrieval System",
        "version": "1.0.0",
        "description": "Intelligent document query system for insurance, legal, HR, and compliance domains",
        "endpoints": {
            "main": "/api/v1/hackrx/run",
            "health": "/health",
            "test": "/test"
        }
    }