"""
Ingest documents: load, chunk, embed, and store in ChromaDB.
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "chroma"
DATA_PATH = "data/"


def ingest():
    # 1. Load documents
    loader = TextLoader(DATA_PATH + "sample.txt")
    documents = loader.load()

    # 2. Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    # 3. Embed and store
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
    db.persist()
    print(f"Stored {len(chunks)} chunks in ChromaDB at {CHROMA_PATH}")


if __name__ == "__main__":
    ingest()
