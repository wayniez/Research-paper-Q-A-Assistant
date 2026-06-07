import fitz
from typing import List, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq

from vectorstore import VectorStore

# --- Class initialization ---
class RAGPipeline:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.vectorstore = VectorStore()
        
    def chunk_text(self, text: str, chunk_size: int = 1500, overlap: int = 150) -> List[str]:
        # make the frame a little smaller, since we're hitting from the side
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        return text_splitter.split_text(text)

    def index_pdf(self, pdf_path: str, filename: str) -> int:
        self.vectorstore.clear()
        all_chunks = []

        with fitz.open(pdf_path) as doc:  # auto-close the document after processing
            for page_num, page in enumerate(doc, 1):
                page_text = page.get_text()
                if not page_text.strip():
                    continue

                page_chunks = self.chunk_text(page_text)
                for chunk in page_chunks:
                    all_chunks.append(f"[Page {page_num}] {chunk}")

        self.vectorstore.add_chunks(all_chunks, filename)
        return len(all_chunks)

    # Change the return type to include the list of chunks used as context
    def ask(self, question: str, chat_history: list, n_results: int=4) -> tuple[str, List[str]]:
        chunks = self.vectorstore.search(question, n_results=n_results)
        context = '\n\n---\n\n'.join(chunks)

        system_prompt = f"""You are an expert research assistant analyzing scientific papers.
Answer questions based ONLY on the provided context from the paper.
If the answer is not in the context, say so clearly. Be precise.

Context:
{context}"""
        MAX_HISTORY = 10 # N of messages to keep in the chat history for context
        trimmed_history = chat_history[-MAX_HISTORY:]
        
        messages = [{'role': 'system', 'content': system_prompt}]
        for msg in trimmed_history:
            messages.append({'role': msg['role'], 'content': msg['content']})
        messages.append({'role': 'user', 'content': question})

        response = self.client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            max_tokens=1000,
            messages=messages
        )
        
        # Return the answer along with the list of chunks that were used as context
        return response.choices[0].message.content, chunks
