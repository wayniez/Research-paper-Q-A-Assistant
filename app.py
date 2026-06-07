__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import tempfile
import streamlit as st

from rag_engine import RAGPipeline


# --- Streamlit UI ---
st.set_page_config(page_title="Scientific Paper RAG", page_icon="📄")
st.title("📄 Scientific Paper Assistant (RAG)")

# State initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag" not in st.session_state:
    # Retrieve the API key from environment variables and initialize the RAG pipeline
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        st.session_state.rag = RAGPipeline(api_key=api_key)
    else:
        st.error("GROQ_API_KEY is not found in environment variables.")

# Sidebar for uploading and indexing PDF
with st.sidebar:
    st.header("1. Upload Paper")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Index Document"):
            with st.spinner("Processing and chunking document..."):
                # Save uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    num_chunks = st.session_state.rag.index_pdf(tmp_file_path, uploaded_file.name)
                    st.success(f"Indexed {num_chunks} chunks successfully!")
                    st.session_state.chat_history = [] # Clear chat history when a new document is indexed
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    os.unlink(tmp_file_path)
    st.divider()
    st.header("2. Settings")
    n_results = st.slider(
        "Context chunks",
        min_value=1,    
        max_value=10,
        value=4,
        help="More chunks = more context, but slower response"
    )

    with st.sidebar:
        st.subheader("🔑 Authentication")
    
    # Field for user to input their own Groq API key, with a note about using the demo key if left blank
    user_key = st.text_input(
        "Groq API Key (Optional)", 
        type="password",
        help="Leave blank to use the developer's demo key. Enter your own if you hit rate limits."
    )
    
    # 1. Determine which API key to use (user-provided or demo)
    if user_key:
        final_key = user_key
        key_source = "user"
    elif "GROQ_API_KEY" in st.secrets:
        final_key = st.secrets["GROQ_API_KEY"]
        key_source = "developer"
    else:
        final_key = None
        key_source = None

    # 2. initialize RAG pipeline with the determined API key, and show status messages
    if final_key:
        # Check if RAG pipeline needs to be initialized or re-initialized (if the key has changed)
        if "rag" not in st.session_state or st.session_state.get("current_key") != final_key:
            st.session_state.rag = RAGPipeline(api_key=final_key)
            st.session_state.current_key = final_key # Remember the current key in session state to avoid unnecessary reinitialization
            
        # provide feedback to the user about which key is being used
        if key_source == "developer":
            st.info("💡 Running on demo API key (limits apply).")
        else:
            st.success("✅ Using your custom API key.")
    else:
        st.error("⚠️ No API key found. Please configure st.secrets or enter a custom Groq key.")

# Main chat interface
st.header("2. Ask Questions")

# Draw chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        # If the message is from the assistant and contains sources, show them in an expander
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("🔍 View text excerpts (Sources):"):
                for idx, chunk in enumerate(message["sources"], 1):
                    st.markdown(f"**Fragment №{idx}**")
                    st.caption(chunk)
                    st.divider()

# Handle user input
if user_query := st.chat_input("Ask a question about the uploaded paper..."):
    # Write user message to the chat and save to history
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    
    # Assistant's response with sources
    with st.chat_message("assistant"):
        with st.spinner("Analyzing the paper..."):
            # Receive both the response and the list of chunks used as sources
            answer, sources = st.session_state.rag.ask(user_query, st.session_state.chat_history[:-1], n_results=n_results)
            
            st.write(answer)
            
            # After displaying the answer, show the sources in an expander
            with st.expander("🔍 View text excerpts (Sources):"):
                for idx, chunk in enumerate(sources, 1):
                    st.markdown(f"**Fragment №{idx}**")
                    st.caption(chunk) # This is where the text starting with [Page X] will go
                    st.divider()
                    
    # Save the assistant's response along with the sources to the chat history
    st.session_state.chat_history.append({
        "role": "assistant", 
        "content": answer, 
        "sources": sources
    })