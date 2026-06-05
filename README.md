# 📄 Scientific Paper Q&A Assistant (RAG)

# 📄 Scientific Paper Q&A Assistant (RAG)

A production-ready, cloud-deployed **Retrieval-Augmented Generation (RAG)** application designed to parse academic papers and scientific PDF publications, enabling users to extract accurate, context-aware answers with transparent source citations.

🚀 **Live Demo:** https://research-paper-q-a-assistant.streamlit.app/

📂 **GitHub Repository:** https://github.com/wayniez/Research-paper-Q-A-Assistant


---

## 🌟 Features

### 📑 PDF Document Processing
Upload and parse complex scientific papers with ease.

### 🧩 Semantic Chunking
Advanced text-splitting pipeline with sliding-window overlap to preserve context across document segments.

### 🔍 Vector Search Engine
Dense semantic retrieval powered by open-source embeddings and local vector indexing.

### 🤖 Context-Aware Question Answering
Fast LLM inference that generates answers exclusively from the uploaded document context.

### ✅ Source Attribution & Factuality
Each response is linked back to relevant document excerpts, improving transparency and reducing hallucinations.

---

## 🛠️ Tech Stack

| Category | Technology |
|-----------|------------|
| **Language** | Python 3.11 |
| **Frontend** | Streamlit |
| **Vector Database** | ChromaDB |
| **Embedding Model** | Hugging Face `all-MiniLM-L6-v2` |
| **LLM** | LLaMA 3 (via Groq API) |
| **Configuration** | TOML |
| **Version Control** | Git |

---

## 📂 Project Structure

```text
├── app.py
├── rag_engine.py
├── vectorstore.py
├── requirements.txt
└── .gitignore
```

### File Overview

| File | Description |
|--------|-------------|
| `app.py` | Streamlit UI and application entry point |
| `rag_engine.py` | Core RAG pipeline, chunking, retrieval, and LLM orchestration |
| `vectorstore.py` | Embedding generation and ChromaDB integration |
| `requirements.txt` | Project dependencies |
| `.gitignore` | Excludes local databases, virtual environments, and temporary files |

---

## ⚡ Local Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/wayniez/Research-paper-Q-A-Assistant.git
cd Research-paper-Q-A-Assistant
```

### 2️⃣ Create a Virtual Environment

**Linux / macOS**

```bash
python -m venv venv
source venv/bin/activate
```

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `httpx==0.27.2` is intentionally pinned to ensure compatibility with the Groq SDK.

### 4️⃣ Configure Secrets

Create the following file:

```text
.streamlit/secrets.toml
```

Add your Groq API key:

```toml
GROQ_API_KEY = "your_actual_groq_api_key_here"
```

### 5️⃣ Run the Application

```bash
streamlit run app.py
```

---

## ☁️ Deployment on Streamlit Cloud

This application is optimized for deployment on **Streamlit Community Cloud**.

### Deployment Steps

1. Push the project to GitHub.
2. Connect the repository to Streamlit Cloud.
3. Set the main file path to:

```text
app.py
```

4. Open **Advanced Settings → Secrets**.
5. Add your API key:

```toml
GROQ_API_KEY = "your_actual_groq_api_key_here"
```

6. Deploy the application.

---

## 🚀 How It Works

```text
PDF Upload
     │
     ▼
Document Parsing
     │
     ▼
Semantic Chunking
     │
     ▼
Embeddings Generation
     │
     ▼
ChromaDB Vector Store
     │
     ▼
Similarity Search
     │
     ▼
Retrieved Context
     │
     ▼
LLaMA 3 (Groq)
     │
     ▼
Answer + Source Citations
```

---

## 📸 Application Workflow

- Upload a scientific paper (PDF).
- The document is parsed and split into semantic chunks.
- Chunks are embedded and stored in ChromaDB.
- User submits a question.
- Relevant context is retrieved from the vector database.
- LLaMA 3 generates an answer based solely on retrieved content.
- Supporting source passages are displayed alongside the answer.

---

## 🔒 Security

- API keys are stored securely using Streamlit Secrets.
- Local vector databases are excluded from version control.
- No document data is permanently stored in the repository.

---

## 📜 License

This project is licensed under the **MIT License**.

Feel free to fork, modify, and use it for personal or commercial projects.
