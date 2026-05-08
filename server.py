"""
server.py — Flask Backend for HR Interview Simulator
Provides API endpoints for RAG-based interview feedback and Q&A.
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate

load_dotenv()

app = Flask(__name__)
CORS(app) # Enable CORS for frontend communication

# ─── Configuration ───────────────────────────────────────────────────────────
CHROMA_PATH = "./chromadb_storage"
COLLECTION  = "hr_rubric"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ─── Initialize RAG Chain ────────────────────────────────────────────────────
def get_rag_chain():
    if not GOOGLE_API_KEY:
        return None
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", 
        google_api_key=GOOGLE_API_KEY
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH, 
        collection_name=COLLECTION, 
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        google_api_key=GOOGLE_API_KEY, 
        temperature=0.4
    )
    
    prompt_template = """You are an elite HR Director. Evaluate the following based on the rubric.
Context: {context}
Input: {question}
Return feedback in clear markdown-like structure with sections: Verdict, Strengths, Gaps, and Score (1-5).
Keep it professional and constructive."""
    
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    return RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever, 
        chain_type_kwargs={"prompt": PROMPT}
    )

chain = get_rag_chain()

# ─── API Endpoints ───────────────────────────────────────────────────────────

@app.route('/api/ask', methods=['POST'])
def ask():
    """General HR Knowledge Q&A"""
    data = request.json
    user_query = data.get('query')
    
    if not chain:
        return jsonify({"error": "AI Engine not initialized. Check API Key."}), 500
    
    try:
        response = chain.invoke({"query": user_query})
        return jsonify({"result": response.get("result")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    """Evaluate an interview answer"""
    data = request.json
    question = data.get('question')
    answer = data.get('answer')
    
    if not chain:
        return jsonify({"error": "AI Engine not initialized. Check API Key."}), 500
    
    full_query = f"Interview Question: {question}\nCandidate Answer: {answer}"
    
    try:
        response = chain.invoke({"query": full_query})
        return jsonify({"result": response.get("result")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
