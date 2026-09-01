import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🤖",
    layout="wide"
)
st.title("🤖 AI Document Assistant & Q&A")
st.caption("Powered by Groq (Llama 3.3) & Streamlit Cloud")

# API Key check from Streamlit Secrets or sidebar
groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

if not groq_api_key:
    groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

if not groq_api_key:
    st.warning("⚠️ Please provide a Groq API Key in secrets or sidebar.")
    st.stop()

# LLM & Embeddings initialization
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    streaming=True
)

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

embeddings = load_embeddings()

# PDF Upload & Vector Store Setup
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Upload PDF here", type=["pdf"])
    
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop("vectorstore", None)
        st.rerun()

    if uploaded_file and "vectorstore" not in st.session_state:
        with st.spinner("Processing PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
            splits = text_splitter.split_documents(docs)

            st.session_state.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings
            )
            os.remove(tmp_path)
            st.success("✅ Document Indexed Successfully!")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_prompt := st.chat_input("Ask a question about the document..."):
    st.chat_message("user").markdown(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        
        if "vectorstore" in st.session_state and st.session_state.vectorstore:
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
            relevant_docs = retriever.invoke(user_prompt)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            prompt_template = f"Context:\n{context}\n\nQuestion: {user_prompt}\nAnswer using the context provided above:"
            response_stream = llm.stream(prompt_template)
        else:
            response_stream = llm.stream(user_prompt)

        for chunk in response_stream:
            full_response += chunk.content
            placeholder.markdown(full_response + "▌")
            
        placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
