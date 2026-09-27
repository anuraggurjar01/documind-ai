from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

def ingest_document(file_path: str):
    """
    Ingests and indexes a document (.pdf, .txt, .md) into the Chroma vector database.
    """
    print(f"Loading {file_path}...")
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in [".txt", ".md"]:
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported formats: .pdf, .txt, .md")

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
        
    print(f"Indexed {len(chunks)} chunks into ChromaDB")
    return vectordb

# Backward compatibility alias
ingest_pdf = ingest_document

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        ingest_document(sys.argv[1])
    else:
        print("Usage: python ingest.py <path_to_file>")