# 🤖 Offline Local AI Assistant with Document RAG

An offline-first, privacy-focused AI Assistant and Document Q&A system powered by local Small Language Models (SLMs) and Vector Search. Built to operate 100% locally without cloud API dependencies.

## 🚀 Key Features
- **Zero Cloud Dependency:** 100% offline inference using local hardware.
- **Local RAG Pipeline:** Ingests and queries PDFs locally using vector embeddings.
- **Data Privacy First:** No user prompts or documents leave the local machine.
- **Streaming UI:** Interactive chat interface with real-time token streaming.

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **LLM Engine:** Ollama (Llama 3.2 3B)
- **Embedding Model:** `nomic-embed-text`
- **Vector Database:** ChromaDB
- **Orchestration:** LangChain
- **Document Processing:** PyPDF

## ⚙️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)<your-username>/offline-local-ai-assistant.git
   cd offline-local-ai-assistant2