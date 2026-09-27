import streamlit as st
from llama_index.core import VectorStoreIndex, Settings, StorageContext, PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb, os

st.set_page_config(page_title="Team RAG Knowledge Assistant", page_icon="📚")
st.title("📚 Team RAG Knowledge Assistant")

@st.cache_resource
def load_query_engine():
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    api_key = os.environ.get("GROQ_API_KEY", "")
    if api_key:
        Settings.llm = Groq(model="llama-3.1-8b-instant", api_key=api_key)
    
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("capstone_knowledge_base")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store)
    
    qa_template = PromptTemplate(
        "You are a helpful assistant that answers ONLY using the context below.\n"
        "If the answer is not contained in the context, say "
        "'I don't have enough information to answer that.'\n\n"
        "Context:\n{context_str}\n\n"
        "Question: {query_str}\n"
        "Answer:"
    )
    return index.as_query_engine(similarity_top_k=3, text_qa_template=qa_template)

try:
    query_engine = load_query_engine()
except Exception as e:
    st.error(f"Error initializing query engine: {e}")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("Ask a question about the knowledge base...")
if question:
    with st.spinner("Searching knowledge base & generating answer..."):
        response = query_engine.query(question)
        sources = sorted({n.metadata.get("file_name", "unknown") for n in response.source_nodes})
        st.session_state.history.append((question, str(response), sources))

for q, a, srcs in st.session_state.history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)
        if srcs:
            st.caption("Sources: " + ", ".join(srcs))
