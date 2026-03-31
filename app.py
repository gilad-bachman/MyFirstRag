from openai import OpenAI
from src.pipeline import create_bachrag_retriever
import streamlit as st
from src.query import query_rag
import os
from dotenv import load_dotenv

load_dotenv()

# --- CACHING THE ENGINE ---
@st.cache_resource
def initialize_bachrag():
    """
    The 'Heart' of the app. Loads everything into RAM once.
    """
    # 1. Initialize the Hybrid Retriever (from pipeline)
    retriever = create_bachrag_retriever()
    
    # 2. Initialize the Groq/OpenAI Client
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_KEY") 
    )
    
    return retriever, client

# Load the heavy machinery
retriever_engine, ai_client = initialize_bachrag()

# --- UI CONFIGURATION ---
st.set_page_config(
    page_title="BachRAG | Gilad Bachman Intel", 
    page_icon="🎼", 
    layout="centered" # 'centered' feels more focused for single-query apps
)

# Custom CSS for a clean, professional "Search" aesthetic
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    /* Style for the Result Card */
    .result-card {
        background-color: #1e2129;
        border-left: 5px solid #4f46e5;
        padding: 25px;
        border-radius: 10px;
        margin-top: 20px;
        color: #e5e7eb;
    }
    .source-tag {
        font-size: 0.8rem;
        color: #9ca3af;
        margin-top: 15px;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🎼 BachRAG")
    st.markdown("---")
    st.markdown("**Version:** 1.0 (Stateless)")
    st.markdown("**Mode:** Single-Query Research")
    st.info("This agent provides direct, factual answers based on Gilad Bachman's documentation. Each query is handled independently.")

# --- MAIN INTERFACE ---
st.title("Gilad Bachman Research Agent")
st.write("Enter a specific inquiry to retrieve factual biographical data.")

# Using a standard text input or chat input without history
prompt = st.chat_input("What would you like to know about Gilad?")

if prompt:
    # We don't save to session_state.messages here. 
    # Instead, we just show the user's active query.
    st.markdown(f"**Researching:** `{prompt}`")
    
    with st.spinner("Accessing ChromaDB..."):
        try:
            # Query the RAG system (sending retriever and client for multi-query expansion and retrieval)
            response = query_rag(prompt, retriever_engine, ai_client)
            
            # Display the result in a professional "Result Card" rather than a chat bubble
            st.markdown(f"""
                <div class="result-card">
                    {response}
                </div>
            """, unsafe_allow_html=True)
            
            # Optional: Add a subtle feedback or action area
            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("New Query"):
                    st.rerun()

        except Exception as e:
            st.error(f"Agent Error: {str(e)}")

else:
    # Placeholder when no query has been made
    st.info("Waiting for input... try asking about Gilad's education or technical stack.")