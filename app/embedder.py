# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS

# def chunk_text(text: str) -> list[str]:
#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
#     return splitter.split_text(text)

# def embed_chunks(chunks: list[str]) -> FAISS:
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     return FAISS.from_texts(chunks, embeddings)

# def search_faiss(vectorstore: FAISS, query: str, k: int = 5):
#     return vectorstore.similarity_search(query, k=k)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import logging
from typing import List, Optional
import re

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Clean and normalize text before chunking."""
    if not text:
        return ""
    
    # Remove excessive whitespace and normalize line breaks
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\"\'\/\%\$\#\@\&\*\+\=\<\>\|\\\n]', ' ', text)
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    return text.strip()


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    """
    Split text into chunks with improved error handling and text cleaning.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between consecutive chunks
    
    Returns:
        List of text chunks
    """
    try:
        if not text or not text.strip():
            logger.warning("Empty or whitespace-only text provided for chunking")
            return []
        
        # Clean the text
        cleaned_text = clean_text(text)
        
        if len(cleaned_text) < 50:
            logger.warning(f"Text too short for effective chunking: {len(cleaned_text)} characters")
            return [cleaned_text] if cleaned_text else []
        
        # Configure text splitter with multiple separators
        separators = [
            "\n\n",  # Double newlines (paragraphs)
            "\n",    # Single newlines
            ". ",    # Sentences
            ", ",    # Clauses
            " ",     # Words
            ""       # Characters (fallback)
        ]
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=separators,
            keep_separator=True
        )
        
        chunks = splitter.split_text(cleaned_text)
        
        # Filter out very small chunks
        filtered_chunks = []
        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) >= 20:  # Minimum chunk size
                filtered_chunks.append(chunk)
            else:
                logger.debug(f"Filtered out small chunk: {len(chunk)} characters")
        
        logger.info(f"Successfully created {len(filtered_chunks)} chunks from {len(cleaned_text)} characters")
        
        if not filtered_chunks:
            logger.warning("No valid chunks created after filtering")
            # Return the original cleaned text as a single chunk if nothing else works
            return [cleaned_text] if cleaned_text else []
        
        return filtered_chunks
        
    except Exception as e:
        logger.error(f"Error during text chunking: {str(e)}")
        # Fallback: return the cleaned text as a single chunk
        try:
            cleaned_text = clean_text(text)
            return [cleaned_text] if cleaned_text else []
        except Exception as fallback_error:
            logger.error(f"Fallback chunking also failed: {str(fallback_error)}")
            return []


def embed_chunks(chunks: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Optional[FAISS]:
    """
    Create embeddings and FAISS vector store from text chunks.
    
    Args:
        chunks: List of text chunks to embed
        model_name: HuggingFace embedding model name
    
    Returns:
        FAISS vector store or None if failed
    """
    try:
        if not chunks:
            logger.error("No chunks provided for embedding")
            return None
        
        # Filter out empty chunks
        valid_chunks = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]
        
        if not valid_chunks:
            logger.error("No valid chunks found after filtering")
            return None
        
        logger.info(f"Creating embeddings for {len(valid_chunks)} chunks using {model_name}")
        
        # Initialize embeddings model with error handling
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},  # Ensure CPU usage for stability
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            logger.error(f"Failed to initialize embedding model {model_name}: {str(e)}")
            # Fallback to a simpler configuration
            embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
        # Create FAISS vector store
        vectorstore = FAISS.from_texts(valid_chunks, embeddings)
        
        logger.info(f"Successfully created FAISS vector store with {len(valid_chunks)} documents")
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error creating embeddings: {str(e)}")
        return None


def search_faiss(vectorstore: FAISS, query: str, k: int = 5) -> List:
    """
    Search FAISS vector store with improved error handling.
    
    Args:
        vectorstore: FAISS vector store
        query: Search query
        k: Number of results to return
    
    Returns:
        List of similar documents
    """
    try:
        if not vectorstore:
            logger.error("No vector store provided for search")
            return []
        
        if not query or not query.strip():
            logger.error("Empty query provided for search")
            return []
        
        # Clean the query
        cleaned_query = clean_text(query)
        
        if not cleaned_query:
            logger.error("Query became empty after cleaning")
            return []
        
        # Perform similarity search
        results = vectorstore.similarity_search(cleaned_query, k=k)
        
        logger.debug(f"Found {len(results)} similar documents for query: {cleaned_query[:50]}...")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during FAISS search: {str(e)}")
        return []


def search_faiss_with_scores(vectorstore: FAISS, query: str, k: int = 5) -> List:
    """
    Search FAISS vector store and return results with similarity scores.
    
    Args:
        vectorstore: FAISS vector store
        query: Search query
        k: Number of results to return
    
    Returns:
        List of (document, score) tuples
    """
    try:
        if not vectorstore:
            logger.error("No vector store provided for search with scores")
            return []
        
        if not query or not query.strip():
            logger.error("Empty query provided for search with scores")
            return []
        
        cleaned_query = clean_text(query)
        
        if not cleaned_query:
            logger.error("Query became empty after cleaning")
            return []
        
        # Perform similarity search with scores
        results = vectorstore.similarity_search_with_score(cleaned_query, k=k)
        
        logger.debug(f"Found {len(results)} similar documents with scores for query: {cleaned_query[:50]}...")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during FAISS search with scores: {str(e)}")
        return []