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
        
        doc = fitz.open(pdf_path)
        all_chunks = []
        
        #   Iterate through each page and extract text, then chunk it
        for page_num, page in enumerate(doc, 1):
            page_text = page.get_text()
            if not page_text.strip():
                continue
                
            # Slicing the page text into chunks
            page_chunks = self.chunk_text(page_text)
            
            # Insert page number into each chunk for better context during retrieval
            for chunk in page_chunks:
                marked_chunk = f"[Page {page_num}] {chunk}"
                all_chunks.append(marked_chunk)
                
        # Add all chunks to the vector store with the filename as part of the ID
        self.vectorstore.add_chunks(all_chunks, filename)
        return len(all_chunks)

    # Change the return type to include the list of chunks used as context
    def ask(self, question: str, chat_history: list) -> tuple[str, List[str]]:
        chunks = self.vectorstore.search(question, n_results=4) # 4 chunks should be enough to provide a good context without overwhelming the model
        context = '\n\n---\n\n'.join(chunks)

        system_prompt = f"""You are an expert research assistant analyzing scientific papers.
Answer questions based ONLY on the provided context from the paper.
If the answer is not in the context, say so clearly. Be precise.

Context:
{context}"""

        messages = [{'role': 'system', 'content': system_prompt}]
        for msg in chat_history:
            messages.append({'role': msg['role'], 'content': msg['content']})
        messages.append({'role': 'user', 'content': question})

        response = self.client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            max_tokens=1000,
            messages=messages
        )
        
        # Return the answer along with the list of chunks that were used as context
        return response.choices[0].message.content, chunks
