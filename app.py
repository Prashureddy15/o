import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(
    page_title="Offline Local AI Assistant", page_icon="🤖", layout="wide"
)
st.title("🤖 Offline Local AI Assistant")
st.caption("100% Offline | Llama 3.2 + Local Document Q&A")

llm = ChatOllama(model="llama3.2:3b", streaming=True)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Sidebar for PDF upload
with st.sidebar:
  st.header("📄 Upload Document")
  uploaded_file = st.file_uploader("Upload PDF here", type=["pdf"])

  if st.button("Clear Chat", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

  if uploaded_file and "vectorstore" not in st.session_state:
    with st.spinner("Processing PDF locally..."):
      with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

      loader = PyPDFLoader(tmp_path)
      docs = loader.load()
      text_splitter = RecursiveCharacterTextSplitter(
          chunk_size=500, chunk_overlap=50
      )
      splits = text_splitter.split_documents(docs)

      st.session_state.vectorstore = Chroma.from_documents(
          documents=splits, embedding=embeddings
      )
      os.remove(tmp_path)
      st.success("✅ PDF Indexed Successfully!")

if "messages" not in st.session_state:
  st.session_state.messages = []

for msg in st.session_state.messages:
  with st.chat_message(msg["role"]):
    st.markdown(msg["content"])

if user_prompt := st.chat_input("Ask a question or inquire about PDF..."):
  st.chat_message("user").markdown(user_prompt)
  st.session_state.messages.append({"role": "user", "content": user_prompt})

  with st.chat_message("assistant"):
    placeholder = st.empty()
    full_response = ""

    if "vectorstore" in st.session_state and st.session_state.vectorstore:
      retriever = st.session_state.vectorstore.as_retriever(
          search_kwargs={"k": 3}
      )
      relevant_docs = retriever.invoke(user_prompt)
      context = "\n\n".join([doc.page_content for doc in relevant_docs])

      prompt_template = f"Context:\n{context}\n\nQuestion: {user_prompt}\nAnswer using the context above:"
      stream = llm.stream(prompt_template)
    else:
      stream = llm.stream(user_prompt)

    for chunk in stream:
      full_response += chunk.content
      placeholder.markdown(full_response + "▌")

    placeholder.markdown(full_response)
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )