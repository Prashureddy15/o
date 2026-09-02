import os
import streamlit as st
from groq import Groq
from pypdf import PdfReader

st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")
st.title("🤖 AI Document Assistant & Q&A")
st.caption("Powered by Groq & Streamlit Cloud")

# Groq API Key
groq_api_key = st.secrets.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY not found in Streamlit Secrets! Please add it in App Settings.")
    st.stop()

# Initialize Groq Client
client = Groq(api_key=groq_api_key)

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
context_text = ""
if uploaded_file is not None:
    with st.spinner("Extracting document text..."):
        reader = PdfReader(uploaded_file)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
        context_text = full_text[:12000]
        st.sidebar.success("Document processed successfully!")

# Display Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        if context_text:
            system_instruction = (
                "You are an assistant for document question-answering. "
                "Use the following context to answer accurately and concisely:\n\n"
                f"{context_text}"
            )
        else:
            system_instruction = "You are a helpful, concise AI assistant."

      try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ],
                model="llama3-8b-8192",
                temperature=0.3
            
            )
            full_response = chat_completion.choices[0].message.content
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            message_placeholder.error(f"Error: {e}")
