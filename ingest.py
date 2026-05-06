"""
ingest.py — Parse the HR rubric document and store it in ChromaDB
Run this ONCE before launching the app:  python ingest.py
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ─── Load environment variables ──────────────────────────────────────────────
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in your .env file.")

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_FILE    = "./data/hr_rubric.txt"
CHROMA_PATH  = "./chromadb_storage"
COLLECTION   = "hr_rubric"

# ─── Step 1: Load the rubric document ────────────────────────────────────────
print(f"[INFO] Loading document: {DATA_FILE}")
loader = TextLoader(DATA_FILE, encoding="utf-8")
documents = loader.load()
print(f"[INFO] Loaded {len(documents)} document(s).")

# ─── Step 2: Split into chunks ───────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""],
)
chunks = splitter.split_documents(documents)
print(f"[INFO] Split into {len(chunks)} chunk(s).")

# ─── Step 3: Embed and store in ChromaDB ─────────────────────────────────────
print("[INFO] Generating embeddings and saving to ChromaDB ...")
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_PATH,
    collection_name=COLLECTION,
)

print(f"[DONE] {len(chunks)} chunk(s) stored in '{CHROMA_PATH}' (collection: '{COLLECTION}').")
print("[DONE] You can now run:  streamlit run app.py")
