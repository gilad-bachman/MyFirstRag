from langchain_chroma import Chroma
from src.pipeline import get_embedding_function
from openai import OpenAI
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import FlashrankRerank
from langchain_classic.retrievers import BM25Retriever, EnsembleRetriever
from langchain_chroma import Chroma
from flashrank import RerankRequest

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



def query_rag(query_text: str, ensemble, client, ranker):
    # --- STEP 1: Multi-Query Expansion ---
    rewrite_prompt = REWRITE_PROMPT_TEMPLATE.format(original_query=query_text)
    rewrite_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": rewrite_prompt}]
    )
    
    queries = [q.strip() for q in rewrite_response.choices[0].message.content.split('\n') if q.strip()]
    queries.append(query_text)
    
    print(f"🔍 Expanding research into {len(queries)} variations...")

    # 2. Fast Retrieval (Accumulate candidates)
    raw_pool = []
    seen_content = set()
    for q in queries:
        docs = ensemble.invoke(q)
        for doc in docs:
            if doc.page_content not in seen_content:
                raw_pool.append(doc)
                seen_content.add(doc.page_content)

    # 3. Optimized Reranking (The Judge runs ONCE)
    pass_to_ranker = [
        {"text": d.page_content, "meta": d.metadata} for d in raw_pool
    ]
    
    # Create the request object that FlashRank expects
    rank_request = RerankRequest(query=query_text, passages=pass_to_ranker)
    
    # Pass the object
    results = ranker.rerank(rank_request)
    
    # Take the top 5 most relevant results
    top_results = results[:5]
    
    # 4. Generate Final Response
    context_text = "\n\n---\n\n".join([r["text"] for r in top_results])
    final_prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant", 
        messages=[{"role": "user", "content": final_prompt}],
        temperature=0
    )
    
    response_text = response.choices[0].message.content.strip()
    sources = sorted(list(set([str(r["meta"].get("id", "Unknown")) for r in top_results])))
    
    return f"{response_text}\n\n**Sources:** {', '.join(sources)}"
        

def main():
    user_query = input("\nQuestion: ")
    print("Searching and generating response...")
    response = query_rag(user_query)
    print("\n" + response)

if __name__ == "__main__":
    main()
