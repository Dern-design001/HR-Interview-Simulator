"""
app.py — HR Interview Simulator (Ultra-Premium UI/UX Edition)
Streamlit frontend powered by Google Gemini + ChromaDB RAG
"""

import os
import time
import random
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Simulator Pro",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed", # Hide sidebar for cleaner UI
)

# ─── Enhanced CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base ──────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #e2e8f0;
}

.stApp {
    background: radial-gradient(circle at top right, #1e1b4b, #0f172a), #0f172a;
    background-attachment: fixed;
}

/* ── Typography ────────────────────────────────────────── */
h1, h2, h3 {
    font-family: 'Outfit', sans-serif !important;
    letter-spacing: -0.02em;
}

.hero-title {
    font-size: 4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #fff 0%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.accent-text {
    color: #818cf8;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 0.8rem;
}

/* ── Layout Containers ─────────────────────────────────── */
.main-container {
    max-width: 1000px;
    margin: 0 auto;
    padding-top: 2rem;
}

/* ── Glass Cards ───────────────────────────────────────── */
.glass-card {
    background: rgba(30, 41, 59, 0.5);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 24px;
    padding: 2.5rem;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    margin-bottom: 2rem;
    transition: transform 0.3s ease, border 0.3s ease;
}

.glass-card:hover {
    border-color: rgba(129, 140, 248, 0.4);
}

.nav-card {
    background: rgba(15, 23, 42, 0.6);
    border-radius: 16px;
    padding: 1rem;
    border: 1px solid rgba(255, 255, 255, 0.05);
    text-align: center;
    cursor: pointer;
    transition: all 0.2s ease;
}

.nav-card:hover {
    background: rgba(129, 140, 248, 0.1);
    border-color: rgba(129, 140, 248, 0.3);
}

/* ── Specific UI Elements ─────────────────────────────── */
.question-box {
    background: linear-gradient(145deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.8));
    border-left: 4px solid #818cf8;
    padding: 1.5rem;
    border-radius: 0 16px 16px 0;
    margin: 1.5rem 0;
}

.score-pill {
    background: rgba(129, 140, 248, 0.2);
    border: 1px solid rgba(129, 140, 248, 0.4);
    color: #a5b4fc;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
}

/* ── Form Inputs ───────────────────────────────────────── */
.stTextArea textarea {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 16px !important;
    color: #f1f5f9 !important;
    padding: 1.2rem !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
}

.stTextArea textarea:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 4px rgba(129, 140, 248, 0.1) !important;
}

/* ── Buttons ───────────────────────────────────────────── */
.stButton button {
    width: 100%;
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    color: white !important;
    border: none !important;
    padding: 0.8rem 2rem !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.4) !important;
}

.stButton button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 20px 25px -5px rgba(79, 70, 229, 0.5) !important;
    filter: brightness(1.1);
}

.stButton button:active {
    transform: translateY(0) !important;
}

/* ── Tab Styling ───────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 2rem;
    background-color: transparent;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.stTabs [data-baseweb="tab"] {
    height: 4rem;
    background-color: transparent !important;
    border: none !important;
    color: #94a3b8 !important;
    font-weight: 600 !important;
}

.stTabs [aria-selected="true"] {
    color: #fff !important;
    border-bottom: 2px solid #818cf8 !important;
}

/* ── Custom Divider ────────────────────────────────────── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
    margin: 2rem 0;
}

/* ── Feedback Section ──────────────────────────────────── */
.feedback-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
}

.feedback-body {
    background: rgba(129, 140, 248, 0.05);
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px dashed rgba(129, 140, 248, 0.3);
    line-height: 1.6;
}

/* ── Sidebar (Hidden but styled) ───────────────────────── */
[data-testid="stSidebar"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)

# ─── Constants & Logic ───────────────────────────────────────────────────────
CHROMA_PATH = "./chromadb_storage"
COLLECTION  = "hr_rubric"

@st.cache_resource(show_spinner=False)
def load_chain():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key: return None, "Missing API Key"
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
        vectorstore = Chroma(persist_directory=CHROMA_PATH, collection_name=COLLECTION, embedding_function=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, temperature=0.4)

        prompt_template = """You are an elite HR Director. Evaluate the following based on the rubric.
Context: {context}
Input: {question}
Return feedback in clear markdown with sections: **Verdict**, **Strengths**, **Gaps**, and **Score (1-5)**."""

        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        def format_docs(docs):
            return "\n\n".join(d.page_content for d in docs)

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | PROMPT
            | llm
            | StrOutputParser()
        )
        return chain, None
    except Exception as e: return None, str(e)

def init_state():
    if "stage" not in st.session_state: st.session_state.stage = "Home"
    if "history" not in st.session_state: st.session_state.history = []
    if "current_q" not in st.session_state: st.session_state.current_q = None

init_state()

# ─── Main Header ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Navigation bar (Custom)
col_logo, col_nav = st.columns([1, 2])
with col_logo:
    st.markdown('<h2 style="margin:0; background: linear-gradient(90deg, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">HR SIM PRO</h2>', unsafe_allow_html=True)

with col_nav:
    nav_cols = st.columns(3)
    if nav_cols[0].button("🏠 Home"): st.session_state.stage = "Home"
    if nav_cols[1].button("🎯 Practice"): st.session_state.stage = "Practice"
    if nav_cols[2].button("🧠 Knowledge"): st.session_state.stage = "Knowledge"

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Load Chain
chain, error = load_chain()
if error:
    st.error(f"Engine Error: {error}")
    st.stop()

# ─── STAGE: HOME ─────────────────────────────────────────────────────────────
if st.session_state.stage == "Home":
    col_text, col_img = st.columns([1.2, 1])
    
    with col_text:
        st.markdown('<p class="accent-text">Next-Gen Interview Intelligence</p>', unsafe_allow_html=True)
        st.markdown('<h1 class="hero-title">Master your next interview.</h1>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:1.2rem; color:#94a3b8; margin-bottom:2rem;">Ground your preparation in real-world HR rubrics. Practice with AI that thinks like a hiring manager.</p>', unsafe_allow_html=True)
        
        if st.button("Get Started Now →"):
            st.session_state.stage = "Practice"
            st.rerun()

    with col_img:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(129, 140, 248, 0.1), rgba(192, 132, 252, 0.1)); border-radius: 32px; padding: 40px; border: 1px solid rgba(255,255,255,0.05); text-align: center;">
            <span style="font-size: 8rem;">💎</span>
            <div style="margin-top: 20px; font-weight: 600; color: #a5b4fc;">RAG-POWERED EVALUATION</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Feature Grid
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        st.markdown('<div class="nav-card"><h3>📈 Scoring</h3><p>Get precise 1-5 rankings based on behavioral traits.</p></div>', unsafe_allow_html=True)
    with f_col2:
        st.markdown('<div class="nav-card"><h3>🛡️ Compliance</h3><p>Learn DEI guidelines and legal boundary questions.</p></div>', unsafe_allow_html=True)
    with f_col3:
        st.markdown('<div class="nav-card"><h3>⚡ Real-time</h3><p>Instant feedback on STAR method articulation.</p></div>', unsafe_allow_html=True)

# ─── STAGE: PRACTICE ─────────────────────────────────────────────────────────
elif st.session_state.stage == "Practice":
    st.markdown('<h1 style="text-align:center;">Interview Sandbox</h1>', unsafe_allow_html=True)
    
    tabs = st.tabs(["🚀 Simulator", "📜 History"])
    
    with tabs[0]:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        if not st.session_state.current_q:
            st.markdown("### Choose your challenge")
            q_cols = st.columns(3)
            if q_cols[0].button("🤝 Behavioral"): st.session_state.current_q = "Tell me about a time you faced a significant challenge at work. How did you handle it?"
            if q_cols[1].button("🧩 Situational"): st.session_state.current_q = "If you were assigned a project with an unclear scope and tight deadline, what would you do?"
            if q_cols[2].button("🌟 Leadership"): st.session_state.current_q = "Tell me about a time you led a project or initiative. What was the outcome?"
        
        if st.session_state.current_q:
            st.markdown('<p class="accent-text">CURRENT QUESTION</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="question-box"><h3 style="margin:0;">{st.session_state.current_q}</h3></div>', unsafe_allow_html=True)
            
            answer = st.text_area("Your Response", placeholder="Use the STAR method...", height=200)
            
            if st.button("ANALYZE RESPONSE"):
                if answer:
                    with st.spinner("Decoding your performance..."):
                        query = f"Question: {st.session_state.current_q}\nCandidate Answer: {answer}"
                        feedback = chain.invoke(query)
                        st.session_state.history.append({"q": st.session_state.current_q, "a": answer, "f": feedback})
                        
                        st.markdown('<div class="feedback-header"><h3>Evaluation Result</h3></div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="feedback-body">{feedback}</div>', unsafe_allow_html=True)
                        
                        if st.button("Next Question"):
                            st.session_state.current_q = None
                            st.rerun()
                else:
                    st.warning("Please provide an answer first.")
            
            if st.button("Skip Question"):
                st.session_state.current_q = None
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

    with tabs[1]:
        if not st.session_state.history:
            st.info("No attempts recorded yet. Start practicing!")
        for i, item in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Attempt {len(st.session_state.history)-i}: {item['q'][:50]}..."):
                st.markdown(f"**Question:** {item['q']}")
                st.markdown(f"**Answer:** {item['a']}")
                st.markdown("---")
                st.markdown(item['f'])

# ─── STAGE: KNOWLEDGE ────────────────────────────────────────────────────────
elif st.session_state.stage == "Knowledge":
    st.markdown('<h1 style="text-align:center;">HR Knowledge Base</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### Query the Rubric")
    user_q = st.text_input("Ask about guidelines, scoring, or DEI...", placeholder="e.g., What are the red flags for teamwork?")
    
    if st.button("CONSULT SYSTEM"):
        if user_q:
            with st.spinner("Searching records..."):
                result = chain.invoke(user_q)
                st.markdown("---")
                st.markdown(result)
    
    st.markdown("---")
    st.markdown("### Common Topics")
    c1, c2, c3 = st.columns(3)
    if c1.button("STAR Method"): 
        st.write(chain.invoke("What is the STAR method and why is it used?"))
    if c2.button("Illegal Questions"): 
        st.write(chain.invoke("What questions are legally prohibited?"))
    if c3.button("Scoring Guide"): 
        st.write(chain.invoke("Explain the 1-5 scoring rubric."))
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align:center; margin-top:5rem; padding-bottom:2rem; color:#475569; font-size:0.8rem;">
    PROTOTYPE V2.0 • BUILT WITH GOOGLE GEMINI & LANGCHAIN • SECURE RAG ARCHITECTURE
</div>
""", unsafe_allow_html=True)
