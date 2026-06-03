# 📄 Paper Chat — RAG Research Assistant

A conversational AI system for analyzing scientific papers 
using Retrieval-Augmented Generation (RAG).

## Tech Stack
- **LLM**: Llama 3.3 70B via Groq API
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector Store**: ChromaDB
- **PDF Processing**: PyMuPDF
- **UI**: Jupyter + ipywidgets

## How it works
1. PDF is extracted and split into overlapping chunks
2. Each chunk is embedded using sentence-transformers
3. On query, top-5 relevant chunks retrieved via cosine similarity
4. Retrieved context + question sent to LLM for answer generation

## Setup
1. Clone the repo
   git clone https://github.com/username/paper-chat.git
   cd paper-chat

2. Install dependencies
   pip install -r requirements.txt

3. Configure API key
   cp .env.example .env
   # open .env.example and paste GROQ_API_KEY

4. Run
   jupyter lab paper_chat.ipynb