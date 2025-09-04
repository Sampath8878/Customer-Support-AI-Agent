# app/index_kb.py
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

loader = DirectoryLoader("kb", glob="*.md")
docs = loader.load()

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(docs, embeddings, persist_directory="chroma_db")
db.persist()

print("✅ KB Indexed successfully!")
