import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")

st.title("🤖 AI Document Assistant & Q&A")
st.caption("Powered by Groq (Llama 3.3) & Streamlit Cloud")

# Groq API Key
groq_api_key = st.secrets.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY not found in Streamlit Secrets! Please add it in App Settings.")
    st.stop()

# Initialize LLM
# Initialize LLM
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    groq_api_key=groq_api_key,
    temperature=0.3
)



# Sidebar for PDF Upload
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Upload PDF here", type=["pdf"])
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Process PDF if uploaded
vectorstore = None
if uploaded_file is not None:
    temp_pdf_path = f"./temp_{uploaded_file.name}"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing document..."):
        loader = PyPDFLoader(temp_pdf_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        st.sidebar.success("Document processed successfully!")
    
    if os.path.exists(temp_pdf_path):
        os.remove(temp_pdf_path)

# Display Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Format documents helper
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# User Input
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        if vectorstore is not None:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            prompt_template = ChatPromptTemplate.from_template(
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer the question. "
                "If you don't know the answer, say that you don't know. Keep answers concise.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            )
            
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt_template
                | llm
                | StrOutputParser()
            )
            full_response = rag_chain.invoke(prompt)
        else:
            response = llm.invoke(prompt)
            full_response = response.content

        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
