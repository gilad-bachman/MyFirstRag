from langchain_chroma import Chroma
from src.pipeline import get_embedding_function
from openai import OpenAI
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import FlashrankRerank
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv


load_dotenv()

groq_api_key = os.getenv("GROQ_KEY")

CHROMA_PATH = "chroma"
DATA_PATH = "./data/"

# Prompt Template to highlight Gilad's strengths while maintaining a professional tone
PROMPT_TEMPLATE = """
You are BachRAG, an objective yet professional biographical assistant for Gilad Bachman. 
Your goal is to provide factual, evidence-based answers that reflect Gilad's career accurately.

CONTEXT FROM DOCUMENTATION:
{context}

---

OPERATING DIRECTIVES:
1. **Fact-First:** Only state what is explicitly written in the context. Do not embellish.
2. **Professional Tone:** Avoid "marketing speak" (e.g., stay away from words like "revolutionary," "amazing," or "unparalleled"). Use a calm, corporate, and precise tone.
3. **The 'Fast Learner' Protocol:** If a specific skill is requested but not found in the context, state: 
   "Current records do not indicate expertise in [Skill], though Gilad's track record shows an aptitude for rapid technical acquisition when expanding his professional scope." 
   (Keep this brief and only use it if the skill is genuinely missing).
4. **No Hallucination:** If the context is empty or irrelevant, simply say you don't have that specific data yet.

USER QUESTION: {question}

RESPONSE:
"""

# Rewrite the user query to be more descriptive and concept-heavy, focusing on Gilad's attainments and competencies
REWRITE_PROMPT_TEMPLATE = """
You are an AI assistant optimizing search queries for a biographical RAG system. 
Generate exactly 3 different versions of the user's original question. 
Each version should use different keywords to ensure high retrieval coverage from a vector database.

User's original question: {original_query}

Provide exactly 3 queries, one per line. Do not include numbers or bullet points.
"""


def get_reranked_retriever():
    # 1. Standard Vector Store Setup
    db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=get_embedding_function()
    )

    # 2. Base Retriever: Fetch MORE candidates than you actually need (e.g., 20)
    base_retriever = db.as_retriever(search_kwargs={"k": 20})

    # 3. Setup the Reranker (Judge)
    compressor = FlashrankRerank()

    # 4. Create the Compression Retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )
    
    return compression_retriever

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=groq_api_key 
)

def query_rag(query_text: str):
    # --- STEP 1: Multi-Query Expansion ---
    rewrite_prompt = REWRITE_PROMPT_TEMPLATE.format(original_query=query_text)
    rewrite_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": rewrite_prompt}]
    )
    # Split the response into a list of 3 queries
    queries = [q.strip() for q in rewrite_response.choices[0].message.content.split('\n') if q.strip()]
    # Add the original query to the list too, just in case!
    queries.append(query_text)
    
    print(f"Expanding research into: {queries}")

    # --- STEP 2: Multi-Retrieval & Deduplication ---
    retriever = get_reranked_retriever()
    all_docs = []
    seen_content = set()

    for q in queries:
        # Retrieve docs for each variation
        docs = retriever.invoke(q)
        for doc in docs:
            # Simple deduplication: don't add the same text twice
            if doc.page_content not in seen_content:
                all_docs.append(doc)
                seen_content.add(doc.page_content)

    # --- STEP 3: Context Preparation ---
    # Now we have a rich, deduplicated list of context chunks
    context_text = "\n\n---\n\n".join([doc.page_content for doc in all_docs])
    
    # --- STEP 4: Final Generation ---
    final_prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant", 
        messages=[{"role": "user", "content": final_prompt}],
        temperature=0
    )
    
    response_text = response.choices[0].message.content.strip()
    sources = list(set([doc.metadata.get("id", "Unknown") for doc in all_docs]))
    
    return f"{response_text}\n\n**Sources:** {', '.join(sources)}"
        

def main():
    user_query = input("\nQuestion: ")
    print("Searching and generating response...")
    response = query_rag(user_query)
    print("\n" + response)

if __name__ == "__main__":
    main()
