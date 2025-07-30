# 🧠 LLM Query–Retrieval System

A FastAPI-based intelligent query engine for large document understanding and clause-based decision making — targeting insurance, legal, HR, and compliance domains.

---

## Features

- PDF, DOCX, and email document parsing
- Clause-level semantic retrieval using FAISS
- LLM-powered decision making with explainability
- Structured JSON responses
- Fast, modular, and production-ready

---

## Tech Stack

- Python
- FastAPI
- OpenAI / Groq LLMs
- LangChain
- FAISS (Vector Search)
- PostgreSQL (planned for doc indexing)

---

## Project Structure
<pre> llm-query-retrieval-system/ │ ├── app/ # All source code lives here │ ├── main.py # Main logic for query handling │ ├── llm_client.py # Groq API interaction logic │ ├── vector_store.py # Embedding + retrieval logic │ └── utils.py # Any reusable functions │ ├── .env # API keys (never push to GitHub) ├── requirements.txt # All Python dependencies ├── README.md # Project overview and usage └── .gitignore # Files to ignore in version control </pre>
