from langchain_chroma import Chroma
from src.pipeline import get_embedding_function
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_KEY")

CHROMA_PATH = "chroma"
DATA_PATH = "./data/"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# New template for query optimization
REWRITE_PROMPT_TEMPLATE = """
You are an AI assistant tasked with optimizing search queries for a RAG system.
The user's original question is: {original_query}

Rewrite this question to be more descriptive and focused on key concepts to improve 
vector database retrieval. Provide ONLY the rewritten query text.
"""

def query_rag(query_text: str):
    # Initialize client
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_api_key 
    )

    # --- STEP 1: Query Rewriting (The "Agent" Call) ---
    rewrite_prompt = REWRITE_PROMPT_TEMPLATE.format(original_query=query_text)
    rewrite_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": rewrite_prompt}]
    )
    optimized_query = rewrite_response.choices[0].message.content.strip()
    print(f"Optimized Query: {optimized_query}") # For debugging

    # --- STEP 2: Retrieval ---
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

    # We search using the OPTIMIZED query
    results = db.similarity_search_with_score(optimized_query, k=5)

    # --- STEP 3: Generation ---
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    # We answer the ORIGINAL question using the retrieved context
    final_prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant", 
        messages=[{"role": "user", "content": final_prompt}]
    )
    
    response_text = response.choices[0].message.content.strip()
    sources = [doc.metadata.get("id", "Unknown") for doc, _score in results]
    
    return f"Response: {response_text}\n\nSources: {sources}"
        

def main():
    user_query = input("\nQuestion: ")
    print("Searching and generating response...")
    response = query_rag(user_query)
    print("\n" + response)

if __name__ == "__main__":
    main()
