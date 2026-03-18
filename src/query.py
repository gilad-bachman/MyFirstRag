from langchain_chroma import Chroma
from src.pipeline import get_embedding_function
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_KEY")
print(groq_api_key)

CHROMA_PATH = "chroma"
DATA_PATH = "./data/"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def query_rag(query_text: str):
    # Load the Chroma vector store and perform a similarity search
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)

    # Create an OpenAI client and generate a response using the prompt
    client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=groq_api_key 
    )
    
    response = client.chat.completions.create(
    model="llama-3.1-8b-instant", 
    messages=[
        {"role": "user", "content": prompt}
        ]
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
