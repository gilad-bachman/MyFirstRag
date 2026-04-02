from flashrank import Ranker
from langchain_classic.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_PATH = "chroma"
DATA_PATH = "./data/"

# Load documents from the data directory
def load_documents():
    print("Loading text documents...")
    # Use TextLoader for .txt files
    loader = DirectoryLoader(
        DATA_PATH, 
        glob="**/*.txt", 
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )
    return loader.load()

# Split the loaded documents into smaller chunks for processing
def split_documents(documents: list[Document]):
    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def get_embedding_function():
    # 'all-MiniLM-L6-v2' is the industry standard for fast, free RAG
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def add_to_chroma(chunks):
    # Load the existing database
    db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=get_embedding_function()
    )

    # 1. Calculate Page IDs (to avoid duplicates)
    chunks_with_ids = calculate_chunk_ids(chunks)

    # 2. Add only new documents
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add chunks that don't exist yet
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("No new documents to add")

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        
        # If page exists, use it; otherwise, just use the source
        if page is not None:
            current_page_id = f"{source}:{page}"
        else:
            current_page_id = source

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Create a cleaner ID
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks

def get_ensemble_retriever(documents: list[Document]):
    # 1. Chroma (Vector Search)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    vector_retriever = db.as_retriever(search_kwargs={"k": 15})

    # 2. BM25 (Keyword Search)
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 5

    # 3. Combine
    return EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5])


def get_reranker_model():
    # This loads the FlashRank model into memory
    return Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="opt/flashrank")

def ingest_data(documents: list[Document]):
    chunks = split_documents(documents)
    add_to_chroma(chunks)

def main():
    print("Warmin up...")
    documents = load_documents() 
    chunks = split_documents(documents)
    add_to_chroma(chunks)

if __name__ == "__main__":
    main()
