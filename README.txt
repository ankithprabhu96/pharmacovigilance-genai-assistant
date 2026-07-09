GenAI Assistant for Pharmacovigilance Case Summarization

A Retrieval-Augmented Generation (RAG) application that summarizes adverse event case reports for pharmacovigilance teams. The assistant retrieves relevant documents from a Chroma vector database and compares responses from multiple LLMs (GPT-4.1-mini, GPT-4.1-nano, Llama 3.2, and Qwen 2.5).

Features

Multi-LLM comparison
Retrieval-Augmented Generation (RAG)
Chroma Vector Database
HuggingFace Embeddings
Streamlit UI
Semantic Search
Similarity Search
Source Traceability
Downloadable Report


Architecture

CSV + JSON

↓

LangChain Documents

↓

Embeddings

↓

ChromaDB

↓

Retriever

↓

RunnableParallel

↓

Multiple LLMs

↓

Streamlit



Installation
python -m venv .venv

pip install -r requirements.txt

Run
python pipeline/create_vector_db.py

streamlit run app.py